import time
from pathlib import Path
from flask import Flask, request
from detect import run
import uuid
import yaml
from loguru import logger
import os
import boto3
import pymongo

images_bucket = os.environ['BUCKET_NAME']

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

app = Flask(__name__)

##new
s3= boto3.client("s3")

@app.route('/predict', methods=['POST'])
def predict():
    # Generates a UUID for this current prediction HTTP request. This id can be used as a reference in logs to identify and track individual prediction requests.
    prediction_id = str(uuid.uuid4())
    logger.info(f'prediction: {prediction_id}. start processing')
    # Receives a URL parameter representing the image to download from S3
    image_name = request.args.get('imgName')
    try:
        original_img_path=image_name
        s3.download_file(images_bucket,image_name,original_img_path)
        #image_local_name = f"/images_path/{img_name}"
        #s3.download_file(images_bucket,img_name,image_local_name)
        #original_img_path = image_local_name

        logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')
    except Exception as e:
        logger.error(f'Failed to download the image :{image_name} .Error:{str(e)}')
        return f'Failed to download image : {image_name}',500


    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='static/data',
        name=prediction_id,
        save_txt=True
    )

    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

    # This is the path for the predicted image with labels
    # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
    predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')
    predicted_img_path_str = str(predicted_img_path)


    try:
        predicted_img="predicted"+image_name
        s3.upload_file(predicted_img_path,images_bucket,predicted_img)
        logger.info(f'prediction: {predicted_img}. Uploading img completed')
    except Exception as e:
        logger.error(f'Failed to upload the predicted image {predicted_img} to s3 .Error:{str(e)}')
        return f'Failed to upload predicted image : {predicted_img}',500

    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')

    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(l[0])],
                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]

        logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'predicted_img_path': predicted_img_path_str,
            'labels': labels,
            'time': time.time()
        }

        ## solution
        replicaSet_name = "myReplicaSet"


        mongo_uri = f"mongodb://mongo1:27017,mongo2:27018,mongo3:27019/?replicaSet={replicaSet_name}"
        try:
            # connect to the MongoDB cluster
            logger.info(f'connecting to the MongoDB cluster')
            client = pymongo.MongoClient(mongo_uri)
        except Exception as e:
            logger.error(f'Failed to connect to MongoDB cluster .Error:str{(e)}')
            return f'Failed', 500

        # The database and collection where we want to store the prediction summaries
        db = client["obejectDetection_db"]
        collection = db["obejectDetection_collection"]

        try:
            collection.insert_one(prediction_summary)
            prediction_summary['_id'] = str(prediction_summary['_id'])
            logger.info(f'prediction Summary stored successfully in MongoDB')
            return prediction_summary
        except Exception as e:
            logger.error(f'Failed to store summary in MongoDB')
            return f'Failed to store summary in MongoDB', 500
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)



