from langchain_google_genai import ChatGoogleGenerativeAI
import torch
import timm
import numpy as np
from dotenv import load_dotenv
import cv2
import os
from langchain.prompts import PromptTemplate
import torch.nn.functional as F

model_path = "models\mushroom_classifier.pth"

class MushroomClassifier:
    def __init__(self):
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        self.class_names = [
            'Agaricus bisporus', 'Agaricus subrufescens', 'Amanita bisporigera', 'Amanita muscaria', 'Amanita ocreata',
            'Amanita phalloides', 'Amanita smithiana', 'Amanita verna', 'Amanita virosa', 'Auricularia auricula-judae',
            'Boletus edulis', 'Cantharellus cibarius', 'Clitocybe dealbata', 'Conocybe filaris', 'Coprinus comatus',
            'Cordyceps sinensis', 'Cortinarius rubellus', 'Entoloma sinuatum', 'Flammulina velutipes', 'Galerina marginata',
            'Ganoderma lucidum', 'Grifola frondosa', 'Gyromitra esculenta', 'Hericium erinaceus', 'Hydnum repandum',
            'Hypholoma fasciculare', 'Inocybe erubescens', 'Lentinula edodes', 'Lepiota brunneoincarnata', 'Macrolepiota procera',
            'Morchella esculenta', 'Omphalotus olearius', 'Paxillus involutus', 'Pholiota nameko', 'Pleurotus citrinopileatus',
            'Pleurotus eryngii', 'Pleurotus ostreatus', 'Psilocybe semilanceata', 'Rhodophyllus rhodopolius', 'Russula emetica',
            'Russula virescens', 'Scleroderma citrinum', 'Suillus luteus', 'Tremella fuciformis', 'Tricholoma matsutake',
            'Truffles', 'Tuber melanosporum'
        ]

        num_classes = len(self.class_names)
        self.model = timm.create_model("rexnet_150", pretrained=True, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key)
        self.prompt_template = PromptTemplate(
            input_variables=["mushroom_name"],
            template=(
                "You are an expert mycologist. When given the name of a mushroom "
                "you will provide the nutritional profile including average protein, fiber, and water content, and state whether it is "
                "safe to consume or not. Do not give mixed opinions; simply state 'safe to eat' or 'not safe to eat' if the mushroom "
                "is poisonous. Mention any possible health risks for unsafe mushrooms. "
                "Classified mushroom: {mushroom_name}."
            )
        )
    def classify_image(self, image_path):
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (224, 224))  
        image_preprocessed = image_resized.astype(np.float32) / 255.0 
        image_tensor = torch.tensor(image_preprocessed).permute(2, 0, 1).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        with torch.no_grad(): 
            predictions = self.model(image_tensor)
            mushroom_class = torch.argmax(F.softmax(predictions, dim=1), dim=1)  
            mushroom_name = self.class_names[mushroom_class.item()] 

        response = self.get_gemini_response(mushroom_name)
        return response

    def get_gemini_response(self, mushroom_name):
        prompt = self.prompt_template.format(mushroom_name=mushroom_name)
        gemini_response = self.gemini_llm.invoke(prompt)
        return gemini_response
