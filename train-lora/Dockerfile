# first image: lightweight alpine to build react project
FROM node:lts-alpine AS build

WORKDIR /app

# copy react project and build
COPY frontend/ .
RUN yarn install
RUN yarn build

# second image: debian to expose services
FROM nginx:stable AS final

WORKDIR /usr/share/nginx/html

# copy react folder "dist" from stage build into nginx image
COPY --from=build /app/dist .
#COPY ./dist .
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# to mount volume if in local
RUN mkdir -p backend/models

# copy backend app into image
RUN mkdir -p backend/train-lora-sdxl-inpaint
COPY backend/app.py backend/my_util.py backend/pipeline.py backend/requirements.txt ./backend/
COPY backend/train-lora-sdxl-inpaint/* backend/train-lora-sdxl-inpaint
# install python, pip, git, redis and pip dependencies
RUN apt-get update && apt-get -y install python3.10 python3-pip git redis-server && pip install --break-system-packages diffusers accelerate transformers peft torch torchvision huggingface_hub flask celery[redis]

RUN ln -s . football-inpaint-train-lora

EXPOSE 80
EXPOSE 5002

# launch redis, celery, flask and nginx
CMD ["sh", "-c", "redis-server --daemonize yes & (cd backend && celery -A pipeline.celery worker --loglevel=INFO) & (cd .. && python3 backend/app.py) & nginx -g 'daemon off;'"]
