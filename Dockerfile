#FROM node:lts-alpine AS build
#
#ARG HTTP_PROXY=
#ARG HTTPS_PROXY=
#ARG NO_PROXY="localhost,127.0.0.1,.localhost"
#ARG http_proxy=
#ARG https_proxy=
#ARG no_proxy="localhost,127.0.0.1,.localhost"
#
#WORKDIR /app
#COPY --exclude=./backend . .
#RUN yarn install
#RUN yarn build

#seconda immagine
FROM nginx:stable-alpine-slim AS final

ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG NO_PROXY="localhost,127.0.0.1,.localhost"
ARG http_proxy=
ARG https_proxy=
ARG no_proxy="localhost,127.0.0.1,.localhost"

WORKDIR /usr/share/nginx/html

#Copy react folder "dist" from stage build into nginx image
#COPY --from=build /app/dist .
COPY ./dist .
COPY nginx.conf /etc/nginx/conf.d/default.conf

#Copy backend app into nginx image
RUN mkdir -p backend/models
COPY backend/app.py backend/my_util.py backend/pipeline.py backend/requirements.txt ./backend
#install python, pip and all other dependencies
RUN apk add python3 py3-pip git --no-cache
#RUN python3 -m venv .venv
RUN python3 -m venv .venv && source .venv/bin/activate && .venv/bin/pip3 install -r backend/requirements.txt

RUN ln -s . football-inpaint

EXPOSE 80
EXPOSE 5000

#launch celery, flask and nginx
CMD ["sh", "-c", "celery -A pipeline.celery worker --loglevel=INFO & python3 backend/app.py & nginx -g 'daemon off;'"]