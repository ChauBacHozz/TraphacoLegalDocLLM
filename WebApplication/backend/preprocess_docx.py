import string
import re
from collections import OrderedDict
from icecream import ic
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

seps = '[.,;-+]'
ascii = string.ascii_lowercase
lowest_level = [(i + ")") for i in ascii]
levels = [lowest_level]
alphabet_lst = [i for i in ascii]
digit_lst = [str(i) for i in range(99)]
bullet_levels1 = [["chương"], ["mục"], ["điều"], digit_lst, alphabet_lst]
bullet_levels2 = [["chương"], ["phụ lục"], digit_lst, alphabet_lst]

def extract_text(doc):
    extracted_text = []
    appendix_index = None
    temp_idx = 0
    for para in doc.paragraphs:
        if para.text != "\xa0":
            extracted_text.append(para.text)
            temp_idx += 1
        if para.alignment == WD_PARAGRAPH_ALIGNMENT.CENTER and "phụ lục" in para.text.lower() and appendix_index == None:
            appendix_index = temp_idx
    return extracted_text, appendix_index

def normalize_bullets(extract_text):
    def check_in_firstn(bullet, end_bullet_idx = 5):
        for i in range(end_bullet_idx):
            if  bullet in bullet_levels1[i]:
                return True, i
        return False, None
    
    text = extract_text
    tree = OrderedDict()

    full_text = []
    c_check = False
    tracking = False
    last_tracking = False
    for i, para in enumerate(text):
        if c_check == True:
            c_check = False
            continue
        if para.split(" ")[0].lower() == "chương":
            para = para + " " + text[i + 1]
            c_check = True
        else:
            c_check = False
        first_token = para.split(" ")[0]
        bullet = "###"
        if len(first_token.strip()) > 0:
            bullet = re.split(r"[.,;)]",first_token)[0]
        # if tracking:

        # if (not check_in_first3(bullet.lower())) and (not tracking):
        #     full_text[-1] = full_text[-1] + ". " + para
        #     if "Thông tư này quy định việc công bố áp dụng và đánh giá việc đáp ứng Thực hành tốt phân phối thuốc, nguyên liệu làm thuốc." in para:
        #         print("CHECKKKKKKKKKKKKKKKKK")
        #     continue
        in_first3, index = check_in_firstn(bullet.lower())
        if tracking and index == 2:
            para = para + " > "

        if in_first3:
            tracking = True
            last_tracking = True
            full_text.append(para)
        else:
            tracking = False
            last_tracking = False
            full_text[-1] = full_text[-1] + ". " + para
        # if (not check_in_first3(bullet.lower())) and (not last_tracking):
        #     # print("Check1")
        #     full_text[-1] = full_text[-1] + " - " + para
        #     if check_in_first3(bullet.lower()):
        #         last_tracking = True
        #     else:
        #         last_tracking = False
        #     continue
        # if (not check_in_first3(bullet.lower())) and (last_tracking):
        #     print("Check2")
        #     full_text[-1] = full_text[-1] + " > " + para
        #     if check_in_first3(bullet.lower()):
        #         last_tracking = True
        #     else:
        #         last_tracking = False
        #     continue
    return full_text

def normalize_appendix_text_bullets(extract_text, appendix_heading_ids):
    def detect_TOC(texts):
        toc = []
        toc_idx = None
        # Loop through texts until duplicate
        for text in texts:
            if text.lower().strip() not in toc:
                toc.append(text.lower().strip())
            else:
                toc_idx = toc.index(text.lower().strip())
                break 
        if len(toc) > 30:
            return []
        return toc[toc_idx:]
    
    def is_heading(text):
        """ Heuristic function to check if a line is a heading """
        if len(text) < 100:  # Headings are usually shorter
            if re.match(r'^[A-Za-z]\)\s+', text):  # Exclude lines starting with 'A)', 'b)', etc.
                return False
            if text[0] in ["-", "(", ")", "+", "*"]:  # Exclude "(a)", "(b)", "(c)"
                return False
            if text.isupper():  # All Caps
                return True
            if text.istitle():  # Title Case
                return True
            if re.match(r'^\d+(\.\d+)*\s+', text):  # Numbered headings (1., 1.1, 2.1.1)
                return True
            if not text.endswith((('.', ':', ',', ';'))):  # No period at the end
                return True
        return False
    
    chunks = []
    apd_size = len(appendix_heading_ids)
    for i in range(apd_size):
        if i < apd_size - 1:
            chunks.append(extract_text[appendix_heading_ids[i]:appendix_heading_ids[i+1]])

        else:
            chunks.append(extract_text[appendix_heading_ids[i]:])
    
    res = []
    for chunk in chunks:
        heading = ": ".join(chunk[:2])
        heading = heading.replace("\n", " ")
        # toc = detect_TOC(chunk)
        bullets = []
        toc = detect_TOC(chunk[2:])
        last_heading = False
        post_heading = False
        for text in chunk[2 + len(toc):]:
            if text in toc and post_heading:
                temp = str(bullets[-1].split(" > ")[:-1])
                bullets.append(temp + " > " + text)
                continue
            if is_heading(text) and text not in toc and len(bullets) > 0 and post_heading and last_heading == False:
                bullets.append(bullets[-1].split(" > ")[0] + " > " + text)
                continue
            # if is_heading(text) and text not in toc and len(bullets) > 0 and last_heading == False:
            #     bullets.append(text)
            #     continue
            if len(bullets) > 0:
                if text.lower() == bullets[0].split(" > ")[0].lower():
                    # The line `toc = bullets[0].split(" > ")` is splitting the first element of the
                    # `bullets` list by the separator " > " and assigning the result to the variable
                    # `toc`. This operation is used to extract individual components from the first
                    # element of the `bullets` list, assuming that the elements are separated by " >
                    # ".
                    # toc = bullets[0].split(" > ")
                    last_heading = True
                    post_heading = False 
                    bullets.append(text)
                    continue
            if is_heading(text) and last_heading == False:
                last_heading = True
                post_heading = False 
                bullets.append(text)
                continue
            if is_heading(text) and last_heading:
                last_heading = True
                post_heading = True
                if bullets[-1].count(">") > 0:
                    bullets.append(" ".join(bullets[-1].split(" > ")[:-1]) + " > " + text)
                else:
                    bullets[-1] = bullets[-1] + " > " + text
                continue

            if is_heading(text) == False and last_heading:
                last_heading = False
                bullets[-1] = bullets[-1] + " > " + text
                continue

            if is_heading(text) == False and last_heading == False:
                last_heading = False
                bullets[-1] = bullets[-1] + "\n" + text
                continue
        for bullet in bullets:
            if len(bullet.strip()) > 0:
                splitter_numbers = bullet.count(">")
                residual = 3 - splitter_numbers
                temp = ""
                for i in range(residual):
                    temp = "> " + temp
                bullet = heading + " " + temp + bullet 
                res.append(bullet)
    print("---------TOC:", toc)
            # break
    return res




    
def check_branch_level(tree):
    level = 0
    next_tree = tree
    # print(isinstance(next_tree, OrderedDict))  # Debugging statement
    while isinstance(next_tree, OrderedDict) and next_tree:  # Check if non-empty OrderedDict
        # Get the last key in the OrderedDict
        last_key = next(reversed(next_tree))
        next_tree = next_tree[last_key]
        level += 1
    return level

def convert_text_list_to_tree(text_list):
    tree = OrderedDict()
    def update_tree(bullet, par):
        for i in range (len(bullet_levels1)):
            if bullet.lower() in bullet_levels1[i]:
                k = tree
                current_branch_level = check_branch_level(k)
                if current_branch_level < i:
                    val = OrderedDict()
                    next_val = val
                    for j in range (i - current_branch_level - 1):
                        next_val[""] = OrderedDict()
                        next_val = next_val[next(reversed(next_val))]
                    next_val[par] = OrderedDict()
                    while isinstance(k, OrderedDict) and k:  # Check if non-empty OrderedDict
                        # Get the last key in the OrderedDict
                        last_key = next(reversed(k))
                        k = k[last_key]
                    k[""] = val
                else:
                    for j in range(i):
                        k = k[next(reversed(k))]
                    k[par] = OrderedDict()
    for i, para in enumerate(text_list):
        first_token = para.split(" ")[0]
        if len(first_token.strip()) > 0:
            bullet = re.split(r"[.,;)]",first_token)[0]
            update_tree(bullet, para)
    return tree

def flatten_tree(tree, parent_path="", separator=" > "):
    flat_list = []
    for key, value in tree.items():
        current_path = f"{parent_path}{separator}{key}" if parent_path else key
        # print(f"Processing: {current_path}")  # Debugging step
        if value:  # Check if the value is not blank OrderedDict
            flat_list.extend(flatten_tree(value, current_path, separator))
        else:
            # If value is a blank ordered dict, convert value to blank string
            flat_list.append((current_path, ""))
    return flat_list

def preprocess_chunks(chunks, heading, doc_number):
    """
    Process raw chunks into a structured format with chapter, section, article, and content.
    """
    processed_chunks = []
    for idx, chunk in enumerate(chunks):
        # Clean up extra whitespace
        chunk = chunk.strip()
        # ic(chunk)
        # Match hierarchical parts using regex
        match = chunk.split(">")
        # ic(match)
        if match:
            if len(match) >= 4:

                chapter = match[0].strip()
                section = match[1].strip() if match[0] else None
                article = match[2].strip()
                # content = match[3].strip()
            else:
                chapter = match[0].strip()
                section = None
                article = None
            content = match[-1].strip()

            if len(content.strip()) == 0:
                print("----------CHECK---------")
                ic(chunk)
            # ic(match.group(4).strip())
        # break

            # Combine hierarchical info with content for embedding
            doc_idx = re.search(f"\d", doc_number)
            if doc_idx:
                doc_id = doc_number[doc_idx.start():]
                
                if "/" in doc_id:
                    doc_id = doc_id.replace("/", "")
                if "-" in doc_id:
                    doc_id = doc_id.replace("-", "")
            else:
                doc_id = idx + 1
            combined_text = doc_id + " > " + chunk
            path = []
            for i in match[:-1]:
                if len(i.strip()) > 0:
                    path.append(i)
            path = ">".join(path)
            # Append structured data
            # content_split = re.split('\s*,\s*.\s*:\s*', content)
            bullet = re.split(r"[.,;)]", content)[0]
             # Ectract Điều 4, Mục 1, khoản ...
            processed_chunks.append({
                "doc_number": doc_number,
                "doc_id": doc_id,          # Unique ID for each chunk
                "heading": heading,
                "middle_path": path,
                "content": content,     # Content text
            })
        else:
            # Handle unmatched chunks (log for review)
            print;(f"Warning: Could not process chunk: {chunk}")
    return processed_chunks