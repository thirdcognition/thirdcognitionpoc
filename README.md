# thirdcognitionpoc
ThirdCognition proof of concept


# Updating base image

```
$ poetry export --without-hashes --format=requirements.txt --output requirements.txt
$ docker build --platform linux/amd64 -t markushaverinen/tc_poc_base . -f Dockerfile_base
$ docker push markushaverinen/tc_poc_base
```

# Building and running the app

```
$ docker build --platform linux/amd64 -t markushaverinen/tc_poc . -f Dockerfile
$ docker run -p 3500:3500 -p 4000:4000 markushaverinen/tc_poc
```