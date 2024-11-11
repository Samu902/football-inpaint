FROM node:lts-alpine AS build

ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG NO_PROXY="localhost,127.0.0.1,.localhost"
ARG http_proxy=
ARG https_proxy=
ARG no_proxy="localhost,127.0.0.1,.localhost"

WORKDIR /app
COPY . .
RUN yarn install
RUN yarn build

#seconda immagine
FROM nginx:stable-alpine-slim AS final

ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG NO_PROXY="localhost,127.0.0.1,.localhost"
ARG http_proxy=
ARG https_proxy=
ARG no_proxy="localhost,127.0.0.1,.localhost"

#install python and flask
RUN apk add python3 py3-flask py3-pillow --no-cache

#con tante dipendenze meglio pip con requirements.txt (pip freeze >)

#Copy folder "dist" from stage build into nginx "image"
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
#Copy flask app.py into nginx image
COPY backend/app.py /usr/share/nginx/html
COPY backend/last_processed.png /usr/share/nginx/html

#!pip install -q -U ultralytics diffusers transformers torch peft pillow opencv-python


WORKDIR /usr/share/nginx/html
RUN ln -s . football-inpaint

EXPOSE 80
EXPOSE 5000

#launch flask and python apps
CMD ["sh", "-c", "flask run --host=0.0.0.0 --port=5000 & nginx -g 'daemon off;'"]