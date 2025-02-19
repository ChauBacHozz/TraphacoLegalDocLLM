
import faiss
import pickle
import os
from icecream import ic
import base64
import hashlib
from neo4j import GraphDatabase
from collections import OrderedDict
import re
from preprocess_docx import (normalize_bullets,
                             convert_text_list_to_tree,
                             flatten_tree,
                             preprocess_chunks)

def save_origin_doc_to_db(new_texts, new_metadata, driver):
    # # Add id to current new_metadata
    def count_nodes(tx):
        query = "MATCH (n) RETURN count(n) AS node_count"
        result = tx.run(query)
        return result.single()["node_count"]
    with driver.session() as session:
        node_count = session.execute_read(count_nodes)
    for i, mtdata in enumerate(new_metadata):
        # bytes_representation = mtdata["path"].encode(encoding="utf-8") 
        # if len(mtdata["content"].strip()) == 0:
        #     print("----------CHECK---------")
        #     ic(mtdata)
        bytes_representation = str(mtdata["doc_id"] + mtdata["middle_path"] + str(node_count + i + 1)).encode("utf-8")
        hash_object = hashlib.sha256(bytes_representation)  # Use SHA-256 (or hashlib.md5 for a smaller hash)
        hash_int = int(hash_object.hexdigest(), 16) % (10**10)
        mtdata["id"] = hash_int
    # # Append new data
    # metadata.extend(new_metadata)

    # # Save the updated FAISS index and metadata
    # save_faiss_and_metadata(index, path_index, data, metadata)

    # # Verify update
    # index, path_index, data, metadata = load_or_initialize_faiss()
    # print(f"📌 Updated FAISS Index: {index.ntotal} entries")
    # print(f"📌 Updated Data: {len(data)} entries")
    # print(f"📌 Updated Metadata: {len(metadata)} entries")
    def create_graph(tx, metadata):
        root_node_content = metadata["heading"]
        root_id = metadata["doc_id"]
        content = metadata["content"]
        id = metadata["id"]
        # Create root node
        tx.run("MERGE (p:Doc_Node:R_Node:Origin_Node {content: $content, d_id: $id})", content = root_node_content,  id = root_id)

        # Create content node, content_bullet = bullet from c_node's content
        
        c_bullet = content.split(" ")[0].rstrip(".,:)")
        if len(c_bullet.split(".")) > 1:
            c_bullet_type = "khoản"
            c_bullet = c_bullet.split(".")[-1]
        else:
            if c_bullet.isalpha():
                c_bullet_type = "điểm"
            else:
                c_bullet_type = "khoản"
        tx.run("MERGE (p:Doc_Node:C_Node:Origin_Node {bullet: $bullet, bullet_type: $bullet_type, content: $content, d_id: $id})", bullet = c_bullet, bullet_type = c_bullet_type, content = content, id = id)

        # Create middle nodes
        middle_node_names = metadata["middle_path"].split(" > ")
        for middle_node in middle_node_names:
            if "chương" in middle_node.lower():
                m_bullet = middle_node.split(" ")[1]
                m_bullet_type = "chương"
            elif "phụ lục" in middle_node.lower():
                m_bullet = middle_node.split(" ")[2].rstrip(",.:)")
                m_bullet_type = "phụ lục"
            else:
                m_bullet = re.split(r"[.,;)]", middle_node)[0]
                if len(m_bullet.split(" ")) > 1:
                    m_bullet_type = m_bullet.split(" ")[0].lower()
                    m_bullet = m_bullet.split(" ")[-1]
                else:
                    if m_bullet.isalpha():
                        m_bullet_type = "điểm"
                    else:
                        m_bullet_type = "khoản"
            tx.run("MERGE (p:Doc_Node:M_Node:Origin_Node {bullet: $bullet, bullet_type: $bullet_type, content: $content, d_id: $id})", bullet = m_bullet, bullet_type = m_bullet_type, content = middle_node, id = root_id)
        # Connect root node to first middle node
        tx.run("""
            MATCH (a:Doc_Node:R_Node:Origin_Node {content: $p_content, d_id: $root_id}), (b:Doc_Node:M_Node:Origin_Node {content: $m_content, d_id: $id})
            MERGE (a)-[:PARENT]->(b)
        """, p_content=root_node_content, m_content=middle_node_names[0], root_id = root_id, id = root_id)
        # Connect last middle node to content node
        tx.run("""
            MATCH (a:Doc_Node:M_Node:Origin_Node {content: $m_content, d_id: $root_id}), (b:Doc_Node:C_Node:Origin_Node {content: $c_content, d_id: $id})
            MERGE (a)-[:CONTAIN]->(b)
        """, m_content=middle_node_names[-1], c_content=content, root_id = root_id, id = id)
        # Connect middle nodes
        for i in range(len(middle_node_names) - 1):
            tx.run("""
                MATCH (a:Doc_Node:M_Node:Origin_Node {content: $node1, d_id: $id}), (b:Doc_Node:M_Node:Origin_Node {content: $node2, d_id: $id})
                MERGE (a)-[:CONTAIN]->(b)
            """, node1=middle_node_names[i], node2=middle_node_names[i + 1], id = root_id)

        # Create nodes
        # for node in nodes:
        #     tx.run("MERGE (n:Node {name: $name})", name=node)

        # # Create relationships
        # for i in range(len(nodes) - 1):
        #     tx.run("""
        #         MATCH (a:Node {name: $node1}), (b:Node {name: $node2})
        #         MERGE (a)-[:NEXT]->(b)
        #     """, node1=nodes[i], node2=nodes[i + 1])
    for mtdata in new_metadata:
        # paths = mtdata["path"].split(">")
        with driver.session() as session:
            session.execute_write(create_graph, mtdata)

def save_modified_doc_to_db(new_texts, new_metadata, driver):
    # # Add id to current new_metadata
    def count_nodes(tx):
        query = "MATCH (n) RETURN count(n) AS node_count"
        result = tx.run(query)
        return result.single()["node_count"]
    with driver.session() as session:
        node_count = session.execute_read(count_nodes)
    for i, mtdata in enumerate(new_metadata):
        # bytes_representation = mtdata["path"].encode(encoding="utf-8") 
        # if len(mtdata["content"].strip()) == 0:
        #     print("----------CHECK---------")
        #     ic(mtdata)
        bytes_representation = str(mtdata["doc_id"] + mtdata["middle_path"] + str(node_count + i + 1)).encode("utf-8")
        hash_object = hashlib.sha256(bytes_representation)  # Use SHA-256 (or hashlib.md5 for a smaller hash)
        hash_int = int(hash_object.hexdigest(), 16) % (10**10)
        mtdata["id"] = hash_int
    def create_tree_paths(text):
        def check_same_type(a, b):
            if a.isdigit() and b.isdigit():
                return True
            elif a.isalpha() and b.isalpha():
                return True
            else:
                return False
        def split_and_concat(text):
            res = re.split("và|,", text)
            new_res = [r.strip() for r in res if r.strip()]
            first_bul = new_res[0].split(" ")[-1]
            fix_bul = "".join(new_res[0].split(" ")[:-1])
            final_res = []
            final_res.append(new_res[0])
            for r in new_res[1:]:
                if check_same_type(r, first_bul):
                    final_res.append(fix_bul + " " + r)
            return final_res
        def flatten_tree(tree, parent_path="", separator=" > "):
            flat_list = []
            for key, value in tree.items():
                current_path = f"{parent_path}{separator}{key}" if parent_path else key
                # print(f"Processing: {current_path}")  # Debugging step
                if value:  # Check if the value is not blank OrderedDict
                    flat_list.extend(flatten_tree(value, current_path, separator))
                else:
                    # If value is a blank ordered dict, convert value to blank string
                    flat_list.append(current_path)
            return flat_list
        pattern = r"\b(?:điểm|khoản|điều|mục|chương)(?:(?!\b(?:điểm|khoản|điều|mục|chương)\b).)*"
        # Building path list
        path = OrderedDict()
        lst = []
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        matches_lst = [match.group().strip().rstrip(",.;") for match in reversed(matches)]
        decompose_matches = []
        # Check if keyword in each elements of matches_lst is valid
        matches_lst_fixed = []
        for match in matches_lst:
            tmp = re.split("điểm|khoản|điều|mục|chương", match, flags=re.IGNORECASE)
            if len(tmp[1].strip().split(" ")[0]) < 4:
                matches_lst_fixed.append(match)
        for match in matches_lst_fixed:
            if "," in match or "và" in match:
                splitted_matcheds = split_and_concat(match)
                for i in splitted_matcheds:
                    decompose_matches.append(i)
            else:
                decompose_matches.append(match)
        d_matches = []
        pattern2 = r"\b(?:điểm|khoản|điều|mục|chương) \w+"
        for i in decompose_matches:
            match = re.search(pattern2, i, re.IGNORECASE)
            if match:
                tmp = match.group().lower()
            else:
                tmp = ""
            d_matches.append(tmp)
        for match in d_matches:
            if match.split(" ")[0].lower() not in lst:
                tmp_path = path
                for i in range(len(lst)):
                    tmp_path = tmp_path[next(reversed(tmp_path))]
                tmp_path[f"{match}"] = OrderedDict()
                lst.append(match.split(" ")[0].lower())
            else:
                index = lst.index(match.split(" ")[0].lower())
                tmp_path = path
                for i in range(index):
                    tmp_path = tmp_path[next(reversed(tmp_path))]
                tmp_path[f"{match}"] = OrderedDict()
        flattened_tree = flatten_tree(path)
        return flattened_tree
    def sub_process_metadata(metadata):
        # Get modified purpose
        modified_content = None
        modified_heading = metadata["content"]
        if "[[" in metadata["content"]:
            modified_heading = metadata["content"].split("[[")[0]
            modified_content = metadata["content"].split("[[")[1].split("]]")[0]

        full_path = metadata["middle_path"] + metadata["content"]

        # Extract modified purpose
        modified_purpose = dict()
        metadata["modified_purpose"] = None
        if "sửa đổi" in full_path.lower():
            modified_purpose["sửa đổi"] = full_path.lower().rfind("sửa đổi")
        if "bổ sung" in full_path.lower():
            modified_purpose["bổ sung"] = full_path.lower().rfind("bổ sung")
        if "bãi bỏ" in full_path.lower():
            modified_purpose["bãi bỏ"] = full_path.lower().rfind("bãi bỏ")

        if modified_purpose:
            metadata["modified_purpose"] = max(modified_purpose, key=modified_purpose.get)
        pattern = r'\d{2,3}/\d{4}/(?:NĐ-CP|TT-CP|TT-BYT|QH\d{2})'

        modified_doc_id = re.search(pattern, full_path)

        if modified_doc_id:
            modified_doc_id = modified_doc_id.group()
            metadata["modified_doc_id"] = modified_doc_id
            if "điều khoản" not in metadata["middle_path"].lower():
                sub_tree = create_tree_paths(modified_heading)
                metadata["modified_paths"] = sub_tree
        else:
            print("Error!!! Cannot find modified document id", full_path)

        if modified_content:
            # Subprocess on modified_content (inside [[]])
            extracted_text = modified_content.split("\n")
            if len(extracted_text) > 1:
                if "chương" in extracted_text[0].lower() and extracted_text[1].isupper():
                    extracted_text[1] = extracted_text[0].strip() + " " + extracted_text[1].strip()
                    extracted_text = extracted_text[1:]

            
            metadata["modified_content"] = extracted_text


    def create_graph(tx, metadata):
        root_node_content = metadata["heading"]
        root_id = metadata["doc_id"]
        content = metadata["content"]
        id = metadata["id"]
        # Create root node
        tx.run("MERGE (p:Doc_Node:R_Node:Modified_Node {content: $content, d_id: $id})", content = root_node_content,  id = root_id)

        # Create content node, content_bullet = bullet from c_node's content
        
        c_bullet = content.split(" ")[0].rstrip(".,:)")
        if len(c_bullet.split(".")) > 1:
            c_bullet_type = "khoản"
            c_bullet = c_bullet.split(".")[-1]
        else:
            if c_bullet.isalpha():
                c_bullet_type = "khoản"
            else:
                c_bullet_type = "mục"
        tx.run("MERGE (p:Doc_Node:C_Node:Modified_Node {bullet: $bullet, bullet_type: $bullet_type, content: $content, d_id: $id})", bullet = c_bullet, bullet_type = c_bullet_type, content = content, id = id)

        # Create middle nodes
        middle_node_names = metadata["middle_path"].split(" > ")
        for middle_node in middle_node_names:
            if "chương" in middle_node.lower():
                m_bullet = middle_node.split(" ")[1]
                m_bullet_type = "chương"
            elif "phụ lục" in middle_node.lower():
                m_bullet = middle_node.split(" ")[2].rstrip(",.:)")
                m_bullet_type = "phụ lục"
            else:
                m_bullet = re.split(r"[.,;)]", middle_node)[0]
                if len(m_bullet.split(" ")) > 1:
                    m_bullet_type = m_bullet.split(" ")[0].lower()
                    m_bullet = m_bullet.split(" ")[-1]
                else:
                    if m_bullet.isalpha():
                        m_bullet_type = "khoản"
                    else:
                        m_bullet_type = "mục"
            tx.run("MERGE (p:Doc_Node:M_Node:Modified_Node {bullet: $bullet, bullet_type: $bullet_type, content: $content, d_id: $id})", bullet = m_bullet, bullet_type = m_bullet_type, content = middle_node, id = root_id)
        # Connect root node to first middle node
        tx.run("""
            MATCH (a:Doc_Node:R_Node:Modified_Node {content: $p_content, d_id: $root_id}), (b:Doc_Node:M_Node:Modified_Node {content: $m_content, d_id: $id})
            MERGE (a)-[:PARENT]->(b)
        """, p_content=root_node_content, m_content=middle_node_names[0], root_id = root_id, id = root_id)
        # Connect last middle node to content node
        tx.run("""
            MATCH (a:Doc_Node:M_Node:Modified_Node {content: $m_content, d_id: $root_id}), (b:Doc_Node:C_Node:Modified_Node {content: $c_content, d_id: $id})
            MERGE (a)-[:CONTAIN]->(b)
        """, m_content=middle_node_names[-1], c_content=content, root_id = root_id, id = id)
        # Connect middle nodes
        for i in range(len(middle_node_names) - 1):
            tx.run("""
                MATCH (a:Doc_Node:M_Node:Modified_Node {content: $node1, d_id: $id}), (b:Doc_Node:M_Node:Modified_Node {content: $node2, d_id: $id})
                MERGE (a)-[:CONTAIN]->(b)
            """, node1=middle_node_names[i], node2=middle_node_names[i + 1], id = root_id)


    # Add properties to metadata before save
    for mtdata in new_metadata:
        # paths = mtdata["path"].split(">")
        sub_process_metadata(mtdata)
    ic(new_metadata[0])
    # ic(new_metadata[:10])
        # with driver.session() as session:
        #     session.execute_write(create_graph, mtdata)

# def save_tree_to_db(tree, driver):
#     def insert_node(tx, parent_name, child_name, child_content, order):
#         """
#         Insert a node and link it to its parent, preserving order.
#         """
#         query = """
#         MERGE (parent:Node {name: $parent_name})
#         MERGE (child:Node {name: $child_name})
#         ON CREATE SET child.content = $child_content, child.order = $order
#         MERGE (parent)-[:HAS_CHILD {order: $order}]->(child)
#         """
#         tx.run(query, parent_name=parent_name, child_name=child_name, child_content=child_content, order=order)

#     def process_dict(tx, parent, data, level=0):
#         """
#         Recursively process the nested ordered dictionary and insert into Neo4j.
#         Maintains order using an "order" property.
#         """
#         for index, (key, value) in enumerate(data.items()):
#             if isinstance(value, OrderedDict):  # If value is a dict, create a node and recurse
#                 insert_node(tx, parent, key, None, index)  # No direct content, just a node
#                 process_dict(tx, key, value, level + 1)  # Recurse deeper
#             else:  # If value is a string, it's the final content
#                 insert_node(tx, parent, key, value, index)

#     # Insert data into Neo4j
#     with driver.session() as session:
#         session.execute_write(process_dict, "Root", tree)  # "Root" as the top-level node

    # print("Data inserted successfully!")

    # Close connection
    # driver.close()
