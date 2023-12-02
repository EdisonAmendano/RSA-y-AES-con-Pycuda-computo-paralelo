FROM node:18-alpine AS dev-deps

WORKDIR /app
COPY package*.json ./
RUN npm install

FROM node:18-alpine AS builder
WORKDIR /app
COPY --from=dev-deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM nginx:alpine AS prod
EXPOSE 80
COPY --from=builder /app/dist/fron2/browser /usr/share/nginx/html

CMD [ "nginx", "-g", "daemon off;" ] esto de angular
