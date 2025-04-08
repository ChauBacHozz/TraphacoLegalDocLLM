import faiss
import pickle
import os
from icecream import ic
import base64
import hashlib
from neo4j import GraphDatabase
from collections import OrderedDict
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from backend.graph_database.preprocessing.preprocess_docx import (normalize_bullets,
                             convert_text_list_to_tree,
                             flatten_tree,
                             preprocess_chunks)

def save_origin_doc_to_db(new_metadata, driver):
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

        
    def create_graph(tx, metadata):
        root_node_content = metadata["heading"]
        d_id = metadata["doc_id"]
        content = metadata["content"]
        c_id = metadata["id"]
        # Create root node
        tx.run("""MERGE (p:Doc_Node:R_Node:Origin_Node {d_id: $d_id})
                  SET p.content = $content""", 
                  content = root_node_content,  d_id = d_id)
        
        # Create middle nodes
        nodes = []
        full_path = ""
        middle_node_names = metadata["middle_path"].split(" > ")
        for middle_node in middle_node_names:
            # Create bullet and bullet type
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
            full_path = full_path + str(m_bullet_type + " " + m_bullet)
            create_node_query = ("""
                OPTIONAL MATCH (n:Origin_Node {d_id: $d_id})
                WHERE $target_path ENDS WITH n.path 
                WITH n, $d_id AS d_id, $path AS path, $content AS content, $bullet AS bullet, $bullet_type AS bullet_type
                CALL apoc.do.when(
                    n IS NULL, 
                    "CREATE (new:Doc_Node:M_Node:Origin_Node {d_id: $d_id, path: $path, content: $content, bullet: $bullet, bullet_type: $bullet_type}) RETURN new", 
                    "SET n.content = $content, n.path = $path RETURN n AS new", 
                    {d_id: d_id, path: path, content: content, bullet: bullet, bullet_type: bullet_type, n: n}
                ) YIELD value
                RETURN value AS node;
            """)
            m_node = tx.run(create_node_query, d_id = d_id, path = full_path, target_path = full_path, content = middle_node, bullet = m_bullet, bullet_type = m_bullet_type)
            m_node = list(m_node)
            # for node_record in m_node:
            #     # Get the node from the record
            #     node = node_record["node"]["new"]
                
            #     # Query to find and delete relationships containing "CONTAIN" in their type
            #     delete_contains_query = """
            #     MATCH (n)-[r]-()
            #     WHERE id(n) = $node_id AND type(r) CONTAINS 'CONTAIN'
            #     DELETE r
            #     """
                
            #     # Execute the query
            #     tx.run(delete_contains_query, node_id=node.id)
            full_path += " > "
            for record in m_node:
                # if "node" in record and record["node"] is not None:
                nodes.append(record["node"]["new"].element_id)


        # Create content node, content_bullet = bullet from c_node's content
        c_bullet = content.split(" ")[0].rstrip(".,:)")
        if len(c_bullet.split(".")) > 1:
            c_bullet_type = "khoản"
            c_bullet = c_bullet.split(".")[-1]
        else:
            if c_bullet.isalpha():
                c_bullet_type = "điểm"
            elif c_bullet.isdigit():
                c_bullet_type = "khoản"
            else:
                c_bullet_type = ""
        full_path = full_path + str(c_bullet_type + " " + c_bullet)

        create_node_query = ("""
            OPTIONAL MATCH (n:Origin_Node {d_id: $d_id})
            WHERE $target_path ENDS WITH n.path 
            WITH n, $d_id AS d_id, $path AS path, $content AS content, $bullet AS bullet, $bullet_type AS bullet_type, $c_id AS c_id
            CALL apoc.do.when(
                n IS NULL, 
                "CREATE (new:Doc_Node:C_Node:Origin_Node {d_id: $d_id, c_id: $c_id, path: $path, content: $content, bullet: $bullet, bullet_type: $bullet_type}) RETURN new", 
                "SET n.content = $content, n.path = $path RETURN n AS new", 
                {d_id: d_id, c_id: c_id, path: path, content: content, bullet: bullet, bullet_type: bullet_type, n: n}
            ) YIELD value
            RETURN value AS node;
        """)
        c_node = tx.run(create_node_query, d_id = d_id, c_id = c_id, path = full_path, target_path = full_path, content = content, bullet = c_bullet, bullet_type = c_bullet_type)
        c_node = list(c_node)
        # Assuming you already have the c_node list from your previous query
        # for node_record in c_node:
        #     # Get the node from the record
        #     node = node_record["node"]["new"]
            
        #     # Query to find and delete relationships containing "CONTAIN" in their type
        #     delete_contains_query = """
        #     MATCH (n)-[r]-()
        #     WHERE id(n) = $node_id AND type(r) CONTAINS 'CONTAIN'
        #     DELETE r
        #     """
            
        #     # Execute the query
        #     tx.run(delete_contains_query, node_id=node.id)
        for record in c_node:
            # if "node" in record and record["node"] is not None:
            nodes.append(record["node"]["new"].element_id)
        # Connect root node to first middle node
        tx.run("""
            MATCH (a:Doc_Node:R_Node:Origin_Node {d_id: $d_id}), (b:Doc_Node:Origin_Node)
            WHERE elementId(b) = $first_middle_node_id
            MERGE (a)-[:CONTAIN]->(b)
            """, d_id = d_id, first_middle_node_id = nodes[0])

        for i in range(len(nodes) - 1):
            tx.run("""
                MATCH (a:Doc_Node:Origin_Node), (b:Doc_Node:Origin_Node)
                WHERE elementId(a) = $firstEId AND elementId(b) = $secondEId
                MERGE (a)-[:CONTAIN]->(b)
            """, firstEId=nodes[i], secondEId=nodes[i + 1])
        
        check_multiple_paths_query = """
            MATCH (a)-[direct:CONTAIN]->(e)
            WITH a, e, direct
            MATCH path = (a)-[*2..]->(e)
            WITH a, e, direct, count(path) AS indirect_path_count
            WHERE indirect_path_count >= 1
            DELETE direct
        """
        tx.run(check_multiple_paths_query)
    for mtdata in new_metadata:
        # paths = mtdata["path"].split(">")
        with driver.session() as session:
            session.execute_write(create_graph, mtdata)

def save_modified_doc_to_db(new_metadata, driver, doc_type = 1):
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
        def flatten_tree(tree, CONTAIN_path="", separator=" > "):
            flat_list = []
            for key, value in tree.items():
                current_path = f"{CONTAIN_path}{separator}{key}" if CONTAIN_path else key
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
                    try:
                        tmp_path = tmp_path[next(reversed(tmp_path))]
                    except:
                        break
                tmp_path[f"{match}"] = OrderedDict()
                lst.append(match.split(" ")[0].lower())
            else:
                index = lst.index(match.split(" ")[0].lower())
                tmp_path = path
                for i in range(index):
                    try:
                        tmp_path = tmp_path[next(reversed(tmp_path))]
                    except:
                        break
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

        black_lst_path = None
        if modified_purpose:
            metadata["modified_purpose"] = max(modified_purpose, key=modified_purpose.get)
        else: 
            metadata["modified_purpose"] = "điều chỉnh"
        pattern = r'\d{2,3}/\d{4}/(?:NĐ-CP|TT-CP|TT-BYT|QH\d{2})'
        if doc_type == 1:
            modified_doc_id = re.search(pattern, full_path)
        else:
            modified_doc_id = metadata["modified_doc_id"]
        # else:
        #     modified_doc_id = 
        if modified_doc_id:
            if doc_type == 1:
                modified_doc_id = modified_doc_id.group()
            metadata["modified_doc_id"] = modified_doc_id
            if "điều khoản" not in metadata["middle_path"].lower():
                sub_trees = create_tree_paths(modified_heading)
                trees = [tree.strip().rstrip(">").strip() for tree in sub_trees]
                metadata["modified_paths"] = trees
            else:
                metadata["modified_paths"] = []
        else:
            # Trường hợp không tham chiếu đến bất kỳ văn bản nào
            metadata["modified_doc_id"] = None
            metadata["modified_paths"] = []

            print("Error!!! Cannot find modified document id", full_path)

        if modified_content:
            # Subprocess on modified_content (inside [[]])
            extracted_text = modified_content.split("\n")
            if len(extracted_text) > 1:
                if "chương" in extracted_text[0].lower() and extracted_text[1].isupper():
                    extracted_text[1] = extracted_text[0].strip() + " " + extracted_text[1].strip()
                    extracted_text = extracted_text[1:]

            # SUBPROCESS FOR CONTENT -> CONVERT TO LIST OF METADATA
            full_text = normalize_bullets(extracted_text)
            # Convert text list to tree base to manage content 
            tree = convert_text_list_to_tree(full_text)
            # Flatten tree into list of strings
            flattened_tree = flatten_tree(tree)
            # Split data into chunks
            chunks = [text[0] for text in flattened_tree]
            # Preprocess chunks
            preprocessed_chunks = preprocess_chunks(chunks, "", modified_doc_id)
            # Extract 'text' atribute from preprocessed_chunks
            texts = [chunk['content'] for chunk in preprocessed_chunks]
            metadata_lst = []
            for chunk in preprocessed_chunks:
                # chunk.pop("content")
                metadata_lst.append(chunk)

            metadata["modified_content"] = metadata_lst
        else:
            metadata["modified_content"] = None

    def create_virtual_origin_nodes(tx, c_node_id, modified_paths, modified_doc_id):
        # Create root node
        tx.run("MERGE (p:Doc_Node:R_Node:Origin_Node {d_id: $root_id})",root_id = modified_doc_id)
        node_order_type = None
        if len(modified_paths) > 0:
            # Create middle nodes if modified_paths exist
            for p in modified_paths:
                # Check if target origin nodes exist or not
                create_node_query = ("""
                    OPTIONAL MATCH (n:Origin_Node {d_id: $d_id})
                    WHERE n.path ENDS WITH $target_path
                    RETURN n ORDER BY n.id
                """)
                result = tx.run(create_node_query, d_id=modified_doc_id, target_path=p)
                res_lst = list(result)
                len_res_lst = len(res_lst)
                if len_res_lst > 0 and res_lst[0]["n"] is not None: 
                    # If exist connect target node to modìied node
                    print("Exist origin node with path:", {res_lst[0]["n"]["path"]})
                    # CẦN SỬA LẠI ĐOẠN NÀY
                    connect_node_query = ("""
                        MATCH (a), (b)
                        WHERE ID(a) = $origin_id AND b.id = $c_node_id
                        MERGE (b)-[:MODIFIED]->(a)
                    """)
                    tx.run(connect_node_query, origin_id = res_lst[0]["n"].id, c_node_id = c_node_id)
                else:
                    path_lst = p.split(" > ")
                    paths = []
                    full_path = ""
                    nodes = []
                    for i, path in enumerate(path_lst):
                        if len(path.split(" ")) > 1:
                            bullet_type = path.split(" ")[0].lower()
                            bullet = path.split(" ")[-1]
                        else:
                            if bullet.isalpha():
                                bullet_type = "khoản"
                            else:
                                bullet_type = "mục"
                        node_order_type = "M_Node"
                        if i == len(path_lst) - 1:
                            node_order_type = "C_Node"
                        full_path += str(bullet_type + " " + bullet)
                        create_node_query = ("""
                            OPTIONAL MATCH (n:Origin_Node {d_id: $d_id})
                            WHERE n.path ENDS WITH $target_path
                            WITH n, $d_id AS d_id, $path AS path, $content AS content, $bullet AS bullet, $bullet_type AS bullet_type
                            CALL apoc.do.when(
                                n IS NULL, 
                                "CREATE (new:Doc_Node:%s:Origin_Node {d_id: $d_id, path: $path, content: $content, bullet: $bullet, bullet_type: $bullet_type}) RETURN new", 
                                "RETURN n AS new", 
                                {d_id: d_id, path: path, content: content, bullet: bullet, bullet_type: bullet_type, n: n}
                            ) YIELD value
                            RETURN value AS node;
                        """ % node_order_type)
                        # create_node_query = f"MERGE (p:Doc_Node:{node_order_type}:Origin_Node" + "{d_id: $root_id, content: $content, bullet: $bullet, bullet_type: $bullet_type, path: $path})"
                        result = tx.run(create_node_query, d_id=modified_doc_id, target_path=path, path=full_path, content=path, bullet=bullet, bullet_type=bullet_type)
                        paths.append(full_path)
                        if node_order_type == "M_Node":
                            full_path += " > "
                    # Connect root node with first middle nodes
                    if len(path_lst) > 1:
                        node_order_type = "M_Node"
                    else:
                        node_order_type = "C_Node"
                    connect_query = (
                        f"MATCH (a:Doc_Node:R_Node:Origin_Node {{d_id: $root_id}}), "
                        f"(b:Doc_Node:{node_order_type}:Origin_Node {{content: $m_content, d_id: $id}}) "
                        "MERGE (a)-[:CONTAIN]->(b)"
                    )
                    tx.run(connect_query, root_id=modified_doc_id, m_content=path_lst[0], id=modified_doc_id)

                    # Connect middle nodes
                    for i in range(len(path_lst) - 1):
                        next_node_type = "M_Node"
                        if i == len(path_lst) - 2:
                            next_node_type = "C_Node"
                        connect_query = (
                            f"MATCH (a:Doc_Node:M_Node:Origin_Node {{content: $node1, d_id: $id, path: $path1}}), "
                            f"(b:Doc_Node:{next_node_type}:Origin_Node {{content: $node2, d_id: $id, path: $path2}}) "
                            "MERGE (a)-[:CONTAIN]->(b)"
                        )
                        tx.run(connect_query, node1=path_lst[i], node2=path_lst[i + 1], id = modified_doc_id, path1 = paths[i], path2 = paths[i+1])                                

                    # Connect last middle node to modified node
                    tx.run("""
                        MATCH (a:Doc_Node:Origin_Node {d_id: $root_id, path: $modified_path}), (b:Doc_Node:C_Node:Modified_Node {id: $id})
                        MERGE (b)-[:MODIFIED]->(a)
                    """, root_id = modified_doc_id, id = c_node_id, modified_path = full_path)
                    if full_path != p:
                        print("ERROR")
                        print("Modified path:", p)
                        print("Full path:", full_path)
                        print("Full path list:", modified_paths)
                        print("Modified content id:", c_node_id)


        else:
            tx.run("""
                MATCH (a:Doc_Node:R_Node:Origin_Node {d_id: $root_id}), (b:Doc_Node:C_Node:Modified_Node {id: $id})
                MERGE (b)-[:MODIFIED]->(a)
            """, root_id = modified_doc_id, id = c_node_id)


    def create_graph(tx, metadata):
        def create_modified_sub_nodes(tx, modified_content, d_id, c_node_id):
            tx.run("MERGE (p:Doc_Node:Sub_Modified_Node:Modified_Node {modified_content: $modified_content, d_id: $d_id})", modified_content = modified_content, d_id = d_id)
            tx.run("""
                    MATCH (p:Doc_Node:Sub_Modified_Node:Modified_Node {modified_content: $modified_content, d_id: $d_id}), (q:Doc_Node:C_Node:Modified_Node {id: $id})
                    MERGE (q)-[:SUB_MODIFIED]->(p)
                   """, modified_content = modified_content, d_id = d_id, id = c_node_id)
        root_node_content = metadata["heading"]
        root_id = metadata["doc_id"]
        content = metadata["content"]

        modified_purpose = metadata["modified_purpose"]
        modified_doc_id = metadata["modified_doc_id"]
        modified_paths = metadata["modified_paths"]
        modified_content = metadata["modified_content"]

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
                c_bullet_type = "điểm"
            else:
                c_bullet_type = "khoản"
        tx.run("MERGE (p:Doc_Node:C_Node:Modified_Node {bullet: $bullet, bullet_type: $bullet_type, content: $content, d_id: $d_id, id: $id, modified_purpose: $modified_purpose})", bullet = c_bullet, bullet_type = c_bullet_type, content = content, id = id, d_id = root_id, modified_purpose = modified_purpose)

        if modified_doc_id:
            create_virtual_origin_nodes(tx, c_node_id=id, modified_paths=modified_paths, modified_doc_id=modified_doc_id)
        if modified_content:
            print(modified_content[0]["content"])
            create_modified_sub_nodes(tx, modified_content[0]["content"], root_id, id)
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
            tx.run("MERGE (p:Doc_Node:M_Node:Modified_Node {bullet: $bullet, bullet_type: $bullet_type, content: $content, d_id: $d_id})", bullet = m_bullet, bullet_type = m_bullet_type, content = middle_node, d_id = root_id)
        # Connect root node to first middle node
        tx.run("""
            MATCH (a:Doc_Node:R_Node:Modified_Node {content: $p_content, d_id: $root_id}), (b:Doc_Node:M_Node:Modified_Node {content: $m_content, d_id: $id})
            MERGE (a)-[:CONTAIN]->(b)
        """, p_content=root_node_content, m_content=middle_node_names[0], root_id = root_id, id = root_id)
        # Connect last middle node to content node
        tx.run("""
            MATCH (a:Doc_Node:M_Node:Modified_Node {content: $m_content, d_id: $root_id}), (b:Doc_Node:C_Node:Modified_Node {content: $c_content, d_id: $root_id})
            MERGE (a)-[:CONTAIN]->(b)
        """, m_content=middle_node_names[-1], c_content=content, root_id = root_id)
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
        with driver.session() as session:
            session.execute_write(create_graph, mtdata)