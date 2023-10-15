from fastapi import FastAPI, File, UploadFile, Response, Depends
import uvicorn
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import datetime



fastapi = FastAPI()



model = YOLO('bestn.pt')
@fastapi.post("/photo")
async def photo(image: UploadFile = File(...)):
    contents = await image.read()
    filename = image.filename
    with open(f"predict.jpg", 'wb') as file:
        file.write(contents)
    return image_n(filename=filename, path_to_photo='predict.jpg')

def image_n(
        path_to_photo,
        filename,
        conf=0.5,
        w_coef=2,
        h_coef=0.25,
) :
    result = model.predict(path_to_photo, conf=conf, save=True)[0]
    nums = [result.names[int(num)] for num in result.boxes.cls]
    probs = result.boxes.conf.tolist()
    xywh = result.boxes.xywhn.tolist()
    data = {tuple(coord): (num, prob) for coord, num, prob in zip(xywh, nums, probs)}

    x_sorted_coords = sorted(xywh, key=lambda x: x[0])
    metadata = [[x_sorted_coords[0]]]

    digits = []



    for x, y, w, h in x_sorted_coords[1:]:
        for arr in metadata:
            last_x, last_y, last_w, last_h = arr[-1]
            if abs(y - last_y) < (h + last_h) * h_coef and abs(x - last_x) < (w + last_w) * w_coef:
                arr.append([x, y, w, h])
                break
        else:
            metadata.append([[x, y, w, h]])


    kord = [[tuple(coord) for coord in arr] for arr in metadata]
    #print(kord[0][1])
    metadata = [[data[tuple(coord)] for coord in arr] for arr in metadata]
    #print(num)

    json_ans = {
        "filename": filename,
        "today_date": datetime.datetime.now()
    }
    ans = []

    for arr in metadata:
        num = ''
        full_prob = 1
        l = 0
        for digit, prob in arr:
            new = {}
            num += digit
            full_prob *= prob
            new['x0'] = kord[0][l][0]
            new['y0'] = kord[0][l][1]
            new['x1'] = kord[0][l][0] + kord[0][l][2]
            new['y1'] = kord[0][l][1]
            new['x2'] = kord[0][l][0] + kord[0][l][2]
            new['y2'] = kord[0][l][1] + kord[0][l][3]
            new['x3'] = kord[0][l][0]
            new['y4'] = kord[0][l][1] + kord[0][l][3]
            new['value'] = digit
            new['confidence'] = prob
            digits.append(new)
            l += 1

        json_ans["confidence"] = prob
        json_ans["full_number"] = num
    json_ans["unrecognized_digits"] = 0
    json_ans['digits'] = digits
    json_ans['position_number'] = {
        'x0' : digits[0]['x0'],
        'y0': digits[0]['y0'],
        'x1': digits[0]['x0'] + digits[-1]['x1'],
        'y1': digits[0]['y0'],
        'x2': digits[0]['x0'] + digits[-1]['x1'],
        'y2': digits[0]['y0'] + digits[-1]['y2'],
        'x3': digits[0]['x0'],
        'y3': digits[0]['y0'] + digits[-1]['y2']
    }



    return json_ans

if __name__ == '__main__':
    uvicorn.run(fastapi, port=9999)