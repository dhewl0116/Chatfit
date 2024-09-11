#app/api/service.py
from ultralytics import YOLO
from typing import Dict, Any
import requests
import base64
import json
import os
import dotenv
import json

dotenv.load_dotenv()


class FoodDetectionService:
    

    @staticmethod
    def load_valid_foods(file_path: str) -> set:
        with open(file_path, 'r') as f:
            food_data = [json.loads(line) for line in f]
        
        valid_foods = set()
        for entry in food_data:
            food_name = entry.get('\ufeff식품명')
            if food_name:
                valid_foods.add(food_name)
        return list(valid_foods)
    
    @staticmethod
    def encode_image_to_base64(image_bytes):
        return base64.b64encode(image_bytes).decode('utf-8')

    @staticmethod
    def initialize_model(model_path: str) -> YOLO:
        return YOLO(model_path)
    
    @staticmethod
    def detect_food_with_gpt(img):
        
        valid_foods = FoodDetectionService.load_valid_foods("app/weight/filtered_food_calories_no_partial_franchise.jsonl")

        base64_image = FoodDetectionService.encode_image_to_base64(img)

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY")
        }



        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"사진 안의 음식들이 어떤 음식인지 알려줘. 그러나 너의 대답은 이 json의 음식 이름 중 하나로만 해야 해. {valid_foods}.\n"
                                "무조건 JSON Format으로 반환해줘\n"
                                "Json_Structure:\n"
                                "{\"detected\": [food1, food2, food3, ....]}\n"
                                "만약 리스트 안에 있는 음식들 중, 내가 준 이미지에 아무 음식도 해당되는게 없으면 {“detected” : []}만 반환해줘"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        content = response.json()['choices'][0]['message']['content']

        start_index = content.find('{')
        end_index = content.rfind('}') + 1
        pure_json_str = content[start_index:end_index]

        try:
            pure_json = json.loads(pure_json_str)
        except json.JSONDecodeError as e:
            print("JSON 변환 중 오류 발생:", e)
            pure_json = {}

        # 응답에서 유효한 음식만 필터링
        detected_foods = pure_json.get('detected', [])
        filtered_foods = [food for food in detected_foods if food in valid_foods]

        return {"detected": filtered_foods}
    
    @staticmethod
    def detect_food(img, model: YOLO) -> Dict[str, Any]:
        results = model.predict(img)

        exception_food_list =  ['Artichoke', 'Banh_trang_tron', 'Banh_xeo', 'Bun_bo_Hue', 'Bun_dau', 'Com_tam', 'Goi_cuon', 'Pho', "Hu_tieu", "Xoi","Vegetable","Squash", "food", "Juice","Cooking-spray","food-drinks","Pear","Baked-goods","Artichoke","Honeycomb", 'Fruit']
        detected_items = []

        for b in results[0].boxes:
            cls = model.names[int(b.cls)]
            if cls not in exception_food_list:
                detected_items.append(cls)

        return {"detected": list(set(detected_items))}
    

