from generate_caption import generate_caption_beam_search, generate_caption_greedy
from flask import Flask, request
import logging

app = Flask(__name__)

@app.route("/")
def error():
    app.logger.info('......................... Someone at "/" {(./.\.)} .........................')
    return "POST at /imgcap/predict/v1 or v2"

@app.route("/imgcap/predict/v1/", methods=['POST'])
def pred_v1():
    app.logger.info('......................... Someone at "../v1/"  {(./.\.)} .........................')
    req_body = request.get_json(force=True)
    b64_img_str = req_body['data']
    caption = generate_caption_greedy(b64_img_str)
    caption = caption[0]
    app.logger.info(f'......................... at v1: {caption} .........................')
    return { 'data' : caption }


@app.route("/imgcap/predict/v2/", methods=['POST'])
def pred_v2():
    app.logger.info('......................... Someone at "../v2/"  {(./.\.)} .........................')
    req_body = request.get_json(force=True)
    b64_img_str = req_body['data']
    caption = generate_caption_beam_search(b64_img_str)
    app.logger.info(f'......................... at v2: {caption} .........................')
    return { 'data' : caption }     

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=80)
    

if __name__ != "__main__":
    app.logger.setLevel(logging.INFO)

app.logger.info("......................... Serving Started .........................")