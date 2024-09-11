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

#Copy folder "dist" from stage build into nginx "image"
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

WORKDIR /usr/share/nginx/html
RUN ln -s . football-inpaint

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]