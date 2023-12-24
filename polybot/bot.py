import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import boto3
import requests


class Bot:

    def __init__(self, token, telegram_chat_url):
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class QuoteBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if msg["text"] != 'Please don\'t quote me':
            self.send_text_with_quote(msg['chat']['id'], msg["text"], quoted_msg_id=msg["message_id"])


class ObjectDetectionBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            try:
                #Download the user photo
                image_path=self.download_user_photo(msg)
                image_name=image_path.split('/')[-1]

                #upload the photo to S3
                images_bucket = os.environ['BUCKET_NAME']
                s3 = boto3.client("s3")
                s3.upload_file(image_path, images_bucket, image_name)
                logger.info(f' {image_name} : Uploading img to s3 completed')
            except Exception as e:
                logger.error(f'Failed to upload {image_name} to S3: {e}')
                return

            try:
                #send a request to the `yolo5` service for prediction
                response = requests.post('http://localhost:8081/predict', params={'imgName': image_name})
                if response.status_code != 200:
                    logger.error(f'the prediction request to yolo5 service failed')
                else:
                    #send results to the Telegram end-user
                    predictions = response.json()
                    logger.info(f'Received predictions: {predictions}')
                    predictionsformat= self.predictions_format(predictions)
                    try:
                        self.send_text(msg['chat']['id'], predictionsformat)
                    except Exception as e:
                        logger.error(f'Failed to send predictions to user: {e}')
            except Exception as e:
                logger.error(f' Requesting predictions from yolo5 service failed !: {e}')


    # find the detected objects and count them in a formatted way
    def predictions_format(self, predictions):
        try:
            labels = predictions.get('labels', [])
            detected_objects = {}
            for label in labels:
                objectDetected = label['class']
                if objectDetected in detected_objects:
                    detected_objects[objectDetected] +=1
                else :
                    detected_objects[objectDetected]=1
            for object, count in detected_objects.items():
                result += f"{object}: {count}\n"
            return "Detected objects:\n"+result

        except Exception as e:
            logger.error(f'prediction format failed: {e}')
            return "Error in predictions_format"
