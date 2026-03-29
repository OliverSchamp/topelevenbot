### OCR


from OOPBot.utils.ocr import PPOCRv5OpenVINO, OCRResult
from pathlib import Path
import cv2
height = 540
width = 960
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

ocr_pipeline = PPOCRv5OpenVINO(
    det_model_path="ocr_model/detector/detector.xml",
    rec_model_path="ocr_model/recognizer/recognizer.xml",
    dict_path="ocr_model/ppocrv5_en_dict.txt"
)

parent = Path("/mnt/SF_NAS/Oliver")

image_path = Path("test.jpg")

img = cv2.cvtColor(cv2.imread(str(parent / image_path)), cv2.COLOR_BGR2RGB)

ocr_result_full_image = ocr_pipeline.run(img)

print([block.text for block in ocr_result_full_image.blocks])

clock_block_list = ocr_result_full_image.find_blocks_by_regex(r"\d{2}:\d{2}")
print(clock_block_list)
clock_block = clock_block_list[0]
to_crop = [0, round(clock_block.box[1][1]) - 25, round(clock_block.box[1][0])-10, height]
img_cropped = img[to_crop[1]:to_crop[3], to_crop[0]:to_crop[2]]

cv2.imwrite(str(parent / "croppedtable.jpg"), cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))

ocr_result_cropped = ocr_pipeline.run(img_cropped)
ocr_result_lines = ocr_result_cropped.sort_into_lines()
print(ocr_result_lines.total_lines)
print([line.line_text for line in ocr_result_lines.lines])


########## Calculate the lines separating every column
## step 1: sort into lines and find the header line
xs = []
ocr_result_lines_full = ocr_result_full_image.sort_into_lines()
for line in ocr_result_lines_full.lines:
    if "pstyl" in line.line_text.lower():
        xs = line.get_word_xs()
        break

from typing import List
def get_fixed_boundaries(points_dict: Dict[str, float], start_bound: int) -> Dict[str, List[float]]:
    # sort the points by their x-coordinate
    points_dict = dict(sorted(points_dict.items(), key=lambda item: item[1]))
    points = list(points_dict.values())
    initial_radius = points[0] - start_bound
    output = {list(points_dict.keys())[0]: [start_bound, points[0] + initial_radius]}
    for k, v in points_dict.items():
        if k == list(points_dict.keys())[0]:
            continue
        prev_point = output[list(output.keys())[-1]][-1]
        current_point = v
        radius = (current_point - prev_point)
        output[k] = [current_point - radius, current_point + radius]
    return output
boundaries = get_fixed_boundaries(xs, 5)
print(boundaries)
# sort ocr_result_lines by boundary that they are inside of

class PlayerInfo(BaseModel):
    name: Optional[str] = None
    roles: List[str] = []
    age: Optional[str] = None
    qlty: Optional[str] = None
    value: Optional[str] = None
    pstyl: Optional[bool] = None
    spec: Optional[str] = None

    VALID_ROLES: List[str] = [
    "GK",   # Goalkeeper
    "DC",   # Center Back
    "DL",   # Left Back
    "DR",   # Right Back
    "DMC",  # Defensive Midfielder
    "AMC",  # Attacking Midfielder
    "AML",  # Left Winger 
    "AMR",  # Right Winger
    "MC",   # Center Midfielder
    "ML",   # Left Midfielder
    "MR",   # Right Midfielder
    "ST"    # Striker
    ]

    def split_roles(self):
        # extract all valid roles from a list of strings. first join the roles into one string. then remove any characters not present in any valid positions. then attempt to construct the remaining string from a combination of concatenated valid positions
        roles_string = "".join(self.roles)
        for char in roles_string:
            if not any(char in role for role in self.VALID_ROLES):
                roles_string = roles_string.replace(char, "")
        extracted_roles = []
        for role in self.VALID_ROLES:
            if role in roles_string:
                extracted_roles.append(role)
                roles_string = roles_string.replace(role, "")
        self.roles = extracted_roles

player_info_list = []
for line in ocr_result_lines.lines:
    new_player_info = PlayerInfo()

    for word in line.words:
        x = word.get_centre_coord()[0]
        for k, v in boundaries.items():
            if v[0] <= x < v[1]:
                if not k.lower() == "roles":
                    setattr(new_player_info, k.lower(), word.text)
                else:
                    new_player_info.roles.append(word.text)
    
    min_y, max_y = line.get_line_y_min_y_max()

    for k, v in boundaries.items():
        if k.lower() == "pstyl":
            pstyl_box = [v[0], min_y, v[1], max_y]
            playstyle_cropped = img_cropped[pstyl_box[1]:pstyl_box[3], pstyl_box[0]:pstyl_box[2]]
            gray = cv2.cvtColor(playstyle_cropped, cv2.COLOR_BGR2GRAY)
            new_player_info.pstyl = (gray < 250).any()


    player_info_list.append(new_player_info)


for player_info in player_info_list:
    player_info.split_roles()
    print(player_info)