# RSA-y-AES-con-Pycuda-computo-paralelo
En esta practica se utilizó algoritmos para la encriptación de texto utilizando lo que es computo paralelo con pycuda en un backend de Flask donde además tenemos unos servicios que se va a consumir desde una aplicación de Angular. Lo que hace la aplicación en pedir al usuario que ingrese un texto al cual se le va a encriptar por el algoritmo que el elija.
#### Ejecución de flask
```
python app.py
```
#### Ejecución de Angular
```
ng serve --o
```
#### Ejecución de DockerCompose
```
docker-compose up
```
### Uso de la Aplicación
#### Servicio para codificar con RSA
http://127.0.0.1:5000/CIRSA, Envía al backend un archivo txt desde el angular que se le pide anteriormente al usuario lo codifica con el algoritmo RSA y retorna un archivo txt que se descarga automáticamente.
#### Servicio para decodificar con RSA
http://127.0.0.1:5000/CIRSA, Envía al backend un archivo txt codificado por el algoritmo RSA desde el angular que se le pide al usuario subirlo lo decodifica con el algoritmo de decodificado de RSA y retorna un archivo txt que se descarga automáticamente.
#### Servicio para codificar con AES
http://127.0.0.1:5000/CIRSA, Envía al backend un archivo txt desde el angular que se le pide anteriormente al usuario lo codifica con el algoritmo AES y retorna un archivo txt que se descarga automáticamente.
#### Servicio para decodificar con AES
http://127.0.0.1:5000/CIRSA, Envía al backend un archivo txt codificado por el algoritmo AES desde el angular que se le pide al usuario subirlo lo decodifica con el algoritmo de decodificado de AES y retorna un archivo txt que se descarga automáticamente.
#### Componentes Principales
* Flask
* Angular
* PyCuda
* Docker
