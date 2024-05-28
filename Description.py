import os

from PIL import Image
from transformers import  AutoProcessor, LlavaForConditionalGeneration
import json
import time
import torch
import torch.nn as nn
#Load model

def extract_description(path,time_b=0):
    time_b=int(time_b)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        # Create a list of devices
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    else:
        # If CUDA is not available, just use CPU
        devices = [torch.device("cpu")]

    model_llava = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model_llava = nn.DataParallel(model_llava, device_ids=devices)

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    prompt = "[SYSTEM] You are an AI assistant specialized in biology and providing accurate and \
    detailed descriptions of animal species.\n<image>\nUSER: You are given the description of an animal species. Provide a very detailed\
    description of the appearance of the species and describe each body part of the animal\
    in detail. Only include details that can be directly visible in a photograph of the\
    animal. Only include information related to the appearance of the animal and nothing\
    else. Make sure to only include information that is present in the species description\
    and is certainly true for the given species. Do not include any information related\
    to the sound or smell of the animal. Do not include any numerical information related\
    to measurements in the text in units: m, cm, in, inches, ft, feet, km/h, kg, lb, lbs.\
    Remove any special characters such as unicode tags from the text. Return the answer as a\
    single paragraph.\nASSISTANT:"

    folders = os.listdir(path)
    path_principal=os.getcwd()
    path2=path.split('/')[-1][:-4]
    if os.path.exists(os.path.join(path_principal, 'Ouputs_LLaVA/'+path2)):
        print("El directorio Ouputs_LLaVA ya existe.")
    else:
        os.mkdir(os.path.join(path_principal, 'Ouputs_LLaVA/'+path2))

    inicio = time.time()
    for folder in folders:
        new_folder=os.path.join(path_principal,'Ouputs_LLaVA/'+path2,folder)
        if os.path.exists(new_folder):
            image_names_old = [nombre[:-5] for nombre in os.listdir(new_folder)]
            image_names_aux = [nombre[:-4] for nombre in os.listdir(os.path.join(path, folder))]
            image_names_aux = list(set(image_names_aux) - set(image_names_old))
            image_names = [nombre + '.jpg' for nombre in image_names_aux]

        else:
            os.mkdir(new_folder)
            image_names = os.listdir(os.path.join(path, folder))


        for img_name in image_names:
           image = Image.open(os.path.join(path,folder,img_name))
           inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
           ## Generate
           generate_ids = model_llava.module.generate(**inputs, max_length=300, min_length=200, do_sample=False)
           description=processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
           _, description = description.split("ASSISTANT:")
           fin=time.time()
           time_1=fin-inicio
           if time_b:
               print('time for prediction: ',time_1)
           #save information
           data = {
               "description": description
           }
           # Guardar el diccionario en un archivo JSON
           json_name=os.path.join(new_folder,img_name[:-4]+'.json')
           with open(json_name, "w") as json_file:
               json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    import os
    import torch
    import json
    from PIL import Image
    import sys
    import torch.nn as nn
    import time

    if len(sys.argv)>3 or len(sys.argv)==1:
        print("Usage: python mi_script.py <path> <time>")
        sys.exit(1)
    path = sys.argv[1]
    if len(sys.argv) == 3 :
        time_b= sys.argv[2]
        extract_description(path, time_b)
    else:
        extract_description(path,time_b=0)


