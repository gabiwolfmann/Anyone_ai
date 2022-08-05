# ay22-01-final-project-3

> Shot Type Classification for Ads

In order to understand the structure of this project, first we need to train a model, and then we create an API to use this model in a production way.

## Train your model 

### Prepare your data

As a first step, we must extract the images from the file `tt0078841.tgz` and put them inside the `data/` folder. Also place the annotations file (`v1_split_trailer.json`) in the same folder. Then, you should be able to run the script `preprocessing/prepare_data_trailer.py`. It will format your data in a way Keras can use for training our CNN model. It should look like this:

```
data/
    ├── v1_full_trailer.json
    ├── v1_split_trailer.json
    ├── data_cleaned
    │   ├── test
    │   │   ├── tt2006051_shot_0009.mp4
    │   │   ├── tt2006051_shot_0010.mp4
    │   │   ├── ...
    │   ├── train
    │   │   ├── tt0444850_shot_0014.mp4
    │   │   ├── tt0444850_shot_0015.mp4
    │   │   ├── ...
    │   ├── val
    │   │   ├── tt2010976_shot_0007.mp4
    │   │   ├── tt2010976_shot_0023.mp4
    │   │   ├── ...
```

### Train 

After we have our images in place, it's time to create our first model and train it on our dataset. To do so, we will make use of `scripts/train_multi_output.py`.

The only input argument it receives is a YAML file with all the experiment settings like dataset, model output folder, epochs,learning rate, etc.

Each time you are going to train a new a model, we create a new folder inside the `experiments/` folder with the experiment name. Inside this new folder, create a `config.yml` with the experiment settings. We also encourage you to store the model weights and training logs inside the same experiment folder to avoid mixing things between different runs. The folder structure should look like this:

```bash
experiments/
    ├── exp_001
    │   ├── config.yml
    │   ├── logs
    │   ├── model.01-6.1625.h5
    │   ├── model.02-4.0577.h5
    │   ├── model.03-2.2476.h5
    │   ├── model.05-2.1945.h5
    │   └── model.06-2.0449.h5
    ├── exp_002
    │   ├── config.yml
    │   ├── logs
    │   ├── model.01-7.4214.h5
    ...
```

The script `scripts/train_multi_output.py` use a video generator for train and validation, that take a batch size of videos, and send them into the model. This script use a multi head in the output in order to classify movement and scale at the same time, but you also can use 
`scripts/train_scale.py` or `scripts/train_movement.py` to run single head experiments.


### Evaluate your trained model

After running many experiments and having a potentially good model trained. It's time to check its performance on our test dataset and prepare a nice report with some evaluation metrics.

We will use the notebook `notebooks/model_evaluation_multi_output.ipynb` to do it.

Its also use a video generator for this task, and use `utils/utils.py` to run the performance evaluation.

### Install

You can use `Docker` to easily install all the needed packages and libraries. 

```bash
$ cd model_train/
$ sudo docker build -t final_project -f Dockerfile .
```

### Run Docker

```bash
$ cd model_train/
$ sudo docker run --rm -p 8888:8888 -it -v $(pwd):/home/app/src --workdir /home/app/src final_project bash
```
or
```bash
$ sudo docker run --rm --net host -it -v $(pwd):/home/app/src --workdir /home/app/src final_project bash
```

### Run jupyter notebook
```bash
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

### Tests

Run:

```bash
$ cd model_train/
$ docker build -t model_train --progress=plain --target test .
```

## Flask ML API

On this part, we will code and deploy an API for serving our own machine learning models. 

The project structure is defined and you will see the modules to understand each proccess.

Below is the full project structure:

```
├── api
│   ├── Dockerfile
│   ├── app.py
│   ├── gunicorn.py
│   ├── input.css
│   ├── middleware.py
│   ├── package-lock.json
│   ├── package.json
│   ├── requirements.txt
│   ├── views.py
│   ├── settings.py
│   ├── utils.py
│   ├── tailwind.config.js
│   ├── templates
│   │   └── index.html
│   ├── static
│   │   ├── app.js
│   │   ├── background.webp
│   │   ├── loading.gif
│   │   └── style.css
│   └── tests
│       ├── test_api.py
│       ├── movie.mp4
│       └── test_utils.py
│
├── model
│   ├── Dockerfile
│   ├── Dockerfile_gpu
│   ├── config_model.yml
│   ├── middleware.py
│   ├── ml_service.py
│   ├── requirements.txt
│   ├── settings.py
│   ├── utils.py
│   ├── weight
│   │   └── model.06-1.5956.h5
│   └── tests
│       ├── test_data
│       │    └── images..
│       └── test_model.py
│
├── proccess_video
│   ├── Dockerfile
│   ├── proccess_video.py
│   ├── requirements.txt
│   ├── scenes_process.py
│   ├── settings.py
│   └── tests
│       ├── test_data
|       |    └── video..
│       └── test_process_video.py
│
├── docker-compose.yml
├── docker-compose.prod.yml
├── README.md
└── tests
    └── test_integration.py
```

Let's take a quick overview on each module:

- api: It has all the needed code to implement the communication interface between the users and our service. It uses Flask and Redis to queue tasks to be processed by our machine learning model.
- model: Implements the logic to get jobs from Redis and process them with our Machine Learning model. When we get the predicted value from our model, we must encole it on Redis again so it can be delivered to the user. This service must comunicate to proccess video service in order to get the list of scenes in a full video.
- process_video: This service take a job from redis and create the logic to split the video in scenes and store in disk 8 frames per each scene of a video. 
- tests: This module contains integration tests so we can properly check our system end-to-end behavior is the expected.

The communication between our services (*api*, *model*, *process_video*) will be done using *Redis*. Every time *api* wants to process a video, it will store the video on disk and send the video name through Redis to the *model* service. *model* already knows in which folder videos are being store, so it only has to use the file name to send this information into process_video service. Here the video will be loaded, splited in each scene and store 8 frames equi-distant correspondingly. Then return to the model service the list of scenes.
For each scene the model predict movement and scale, and after predict all these scenes return the results back to *api*.


### Install and run

To run the services using compose:

```bash
$ docker-compose up --build -d
```
If you want to run it in some server with GPU support you have to use dockerfile_gpu instead.
So the command must be:
```bash
$ docker compose -f docker-compose.yml -f docker-compose.prod.yml  up --build -d
```
The second yml will override which dockerfile have to use it.

To stop the services:

```bash
$ docker-compose down
```

### Tests

#### Integration end-to-end

You must have the full pipeline running and [requests](https://docs.python-requests.org/en/latest/) library installed. Then, from this project root folder run:

```
$ python tests/test_integration.py
```

#### Api

Run:

```bash
$ cd api/
$ docker build -t flask_api_test --progress=plain --target test .
```

#### Model

Same as api, run:

```bash
$ cd model/
$ docker build -t model_test --progress=plain --target test .
```

#### Process Video

```bash
$ cd process_video/
$ docker build -t process_video --progress=plain --target test .
```