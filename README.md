# OCR For Low Resource Languages

This work is an adaptation of CNN+Transformer architecture to training text recognition models for Yorùbá & Igbo.

## Architecture


![alt text](samples/architecture.png)

## Setup

- Clone the github repository
    ```
    $ git clone https://github.com/ToluClassics/LowResourceOCR.git
    ```
- Create a virtual environment and install dependencies
    ```
    $ python2 -m venv venv 
    $ (venv) pip install -r requirements.txt
    ```
- Download TextRecognition Model from google drive
    - Igbo : [Download link](https://drive.google.com/file/d/14YujZltsPMIkxnikq9ZdfQwc72aNX5dN/view?usp=sharing)
    - Yorùbá : [Download link](https://drive.google.com/file/d/17KnBn1cwH4scDaC36-lCZ-t1bHwmWCL3/view?usp=sharing)
    - English : [Download link](https://drive.google.com/file/d/17KnBn1cwH4scDaC36-lCZ-t1bHwmWCL3/view?usp=sharing)

- Run Inference:

    ![alt text](samples/yor_sample.jpg)

    ```
    python3 inference.py --lang yor 
        --image_path samples/yor_sample.jpg 
        --checkpoint_path run/checkpoint_weights_igbo_trdg.pt
    ```
    #### Output

    ```
    [INFO] Load pretrained model
    [INFO] Predicted text is: Àwọn Ohun Tó Wà
    ```

## Reference::

-  Scene Text Recognition via Transformer, [Paper](https://arxiv.org/abs/2003.08077)
