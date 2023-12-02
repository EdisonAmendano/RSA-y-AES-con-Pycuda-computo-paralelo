from flask import Flask, request, send_file
from flask_cors import CORS
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
from pycuda import gpuarray
import numpy as np
import math
import time

app = Flask(__name__)
CORS(app)

@app.route('/CIRSA', methods=['POST'])
def CifradoRSA():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Leemos los datos del archivo en memoria
        file_data = file.read().decode('utf-8')
        # Pasamos los datos directamente a la función AES
        decrypted_file_path = RSA(file_data)

        

        return send_file(decrypted_file_path, as_attachment=True)
    return 'Something went wrong'


def RSA(mensaje_original):
    e = 17
    n = 3233

    # Cifrar el mensaje
    mensaje_cifrado = encriptar(mensaje_original, e, n)
    # Guardar el mensaje cifrado en un archivo
    with open("mensaje_cifradoCuda.txt", "w", encoding="utf-8") as file:
        file.write(" ".join(map(str, mensaje_cifrado)))
    path= "mensaje_cifradoCuda.txt"
    return path

#Cifrado

def conversion_ascii(texto):
    return np.array([ord(char) for char in texto], dtype=np.int32)

def encriptar(texto, e=17, n=3233):
    cuda.init()
    device = cuda.Device(0)
    context = device.make_context()
    try:
        texto_np = conversion_ascii(texto)
        texto_gpu = gpuarray.to_gpu(texto_np)

        encriptacion_rsa = """
            __device__ int exp_mod(int base, int exp, int mod) {
                int res = 1;
                base = base % mod;
                while (exp > 0) {
                    if (exp & 1)
                        res = (res * base) % mod;
                    exp >>= 1;
                    base = (base * base) % mod;
                }
                return res;
            }

            __global__ void encriptacion_rsa(int *valores, int clave, int modulo, int tam) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < tam) {
                    valores[i] = exp_mod(valores[i], clave, modulo);
                }
            }
        """

        mod = SourceModule(encriptacion_rsa)
        rsa_kernel = mod.get_function("encriptacion_rsa")

        hilos_bloque = 256
        bloques = math.ceil(len(texto_np) / hilos_bloque)

        inicio = time.time()  # Registra el tiempo de inicio

        rsa_kernel(texto_gpu, np.int32(e), np.int32(n), np.int32(len(texto_np)), block=(hilos_bloque, 1, 1), grid=(bloques, 1))

        context.synchronize()

        fin = time.time()  # Registra el tiempo de finalización

        resultado = texto_gpu.get()
        tiempo_ejecucion = (fin - inicio) * 1000  # Convierte a milisegundos
        print(f"Tiempo de ejecución: {tiempo_ejecucion:.6f} milisegundos")
   
        return resultado.tolist()
    finally:
        context.pop()
        context.detach()

#Decifrado

@app.route('/DERSA', methods=['POST'])
def DesifradoRSA():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Leemos los datos del archivo en memoria
        mensaje_cifrado_str = file.read()
        mensaje_cifrado = [int(num) for num in mensaje_cifrado_str.split()]
        # Pasamos los datos directamente a la función AES
        archivo_descifrado = decifrarRSA(mensaje_cifrado)

        

        return send_file(archivo_descifrado, as_attachment=True)
    return 'Something went wrong'

def decifrarRSA(mensaje_cifrado):
    d = 2753
    n = 3233

    # Descifrar el mensaje cifrado
    mensaje_descifrado = desencriptar(mensaje_cifrado, d, n)

    # Guardar el mensaje descifrado en un archivo
    archivo_descifrado = "mensaje_descifradoCuda.txt"
    with open(archivo_descifrado, "w", encoding="utf-8") as file:
        file.write(mensaje_descifrado)

    return archivo_descifrado

def conversion_texto(lista_ascii):
    return ''.join(chr(num) for num in lista_ascii)

def desencriptar(texto_encriptado, d=2753, n=3233):
    cuda.init()
    device = cuda.Device(0)
    context = device.make_context()
    try:
        texto_encriptado_np = np.array(texto_encriptado, dtype=np.int32)
        texto_encriptado_gpu = gpuarray.to_gpu(texto_encriptado_np)

        desencriptacion_rsa = """
            __device__ int exp_mod(int value, int power, int mod) {
                int result = 1;
                value = value % mod;
                while (power > 0) {
                    if (power & 1)
                        result = (long long)result * value % mod;
                    power >>= 1;
                    value = (long long)value * value % mod;
                }
                return result;
            }

            __global__ void descencriptacion_rsa(int *values, int exponent, int modulus, int length) {
                int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < length) {
                    values[index] = exp_mod(values[index], exponent, modulus);
                }
            }
        """
        mod = SourceModule(desencriptacion_rsa)
        rsa_kernel = mod.get_function("descencriptacion_rsa")

        hilos_bloque = 256
        bloques = math.ceil(len(texto_encriptado) / hilos_bloque)

        inicio = time.time()  # Registra el tiempo de inicio

        rsa_kernel(texto_encriptado_gpu, np.int32(d), np.int32(n), np.int32(len(texto_encriptado)),
                   block=(hilos_bloque, 1, 1), grid=(bloques, 1))

        context.synchronize()

        fin = time.time()  # Registra el tiempo de finalización

        resultado = texto_encriptado_gpu.get()

        tiempo_ejecucion = (fin - inicio) * 1000  # Convierte a milisegundos
        print(f"Tiempo de ejecución: {tiempo_ejecucion:.6f} milisegundos")

        texto_desencriptado = conversion_texto(resultado)
        return texto_desencriptado
    finally:
        context.pop()
        context.detach()

#AES

# Función para cifrar usando PyCUDA
def encrypt_with_dummy_pycuda(file_data, dummy_encrypt):
    # Determinar la cantidad de datos y crear un buffer para ellos
    data_size = len(file_data)
    data_gpu = cuda.mem_alloc(data_size)
    cuda.memcpy_htod(data_gpu, file_data)

    # Calcular tamaños de grilla y bloque
    block_size = 256
    grid_size = (data_size + block_size - 1) // block_size

    # Lanzar el kernel en la GPU
    dummy_encrypt(data_gpu, np.int32(data_size), block=(block_size, 1, 1), grid=(grid_size, 1))

    # Crear un buffer para los datos cifrados y copiarlos de vuelta a la CPU
    encrypted_data = cuda.pagelocked_empty(data_size, dtype=np.uint8)
    cuda.memcpy_dtoh(encrypted_data, data_gpu)

    # Liberar memoria de la GPU
    data_gpu.free()

    return encrypted_data

# Función para leer un archivo y cifrar su contenido
def encrypt_file(file_data, dummy_encrypt):

    # Comienza a medir el tiempo
    start_time = time.time()

    # Cifrar el contenido del archivo
    encrypted_data = encrypt_with_dummy_pycuda(file_data, dummy_encrypt)

    # Termina de medir el tiempo
    end_time = time.time()

    # Escribir los datos cifrados de vuelta a un archivo nuevo
    encrypted_file_path = "AES" + ".txt"
    with open(encrypted_file_path, 'wb') as encrypted_file:
        encrypted_file.write(encrypted_data)

    # Calcula la duración del proceso de cifrado
    encryption_duration = end_time - start_time

    return encrypted_file_path


def AES(file_path):
    cuda.init()
    device = cuda.Device(0)
    context = device.make_context()
    aes_cuda_kernel = """
    __global__ void dummy_encrypt(unsigned char *data, int data_size) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < data_size) {
            // Operación de cifrado dummy (XOR con 0xFF)
            data[idx] = data[idx] ^ 0xff;
        }
    }
    """

    # Compilar el código del kernel
    module = SourceModule(aes_cuda_kernel)
    dummy_encrypt = module.get_function("dummy_encrypt")
    encrypted_file_path= encrypt_file(file_path, dummy_encrypt)
    context.pop()
    context.detach()
    return encrypted_file_path


@app.route('/CiAES', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Leemos los datos del archivo en memoria
        file_data = file.read()
        # Pasamos los datos directamente a la función AES
        encrypted_file_path = AES(file_data)
        return send_file(encrypted_file_path, as_attachment=True)
    return 'Something went wrong'

  



# Función para descifrar usando PyCUDA, que es esencialmente la misma que la función de cifrado
def decrypt_with_dummy_pycuda(file_data, dummy_encrypt):
    # El proceso de descifrado es idéntico al proceso de cifrado debido a la naturaleza de XOR
    return encrypt_with_dummy_pycuda(file_data, dummy_encrypt)

# Función para descifrar un archivo
def decrypt_file(file, dummy_encrypt):

    # Comienza a medir el tiempo
    start_time = time.time()

    # Descifrar el contenido del archivo
    decrypted_data = decrypt_with_dummy_pycuda(file, dummy_encrypt)

    # Termina de medir el tiempo
    end_time = time.time()

    # Escribir los datos descifrados de vuelta a un archivo nuevo
    decrypted_file_path = "AES_DEC2" + ".txt"
    with open(decrypted_file_path, 'wb') as decrypted_file:
        decrypted_file.write(decrypted_data)

    # Calcula la duración del proceso de descifrado
    decryption_duration = end_time - start_time

    return decrypted_file_path

def AESDES(file_data):
    cuda.init()
    device = cuda.Device(0)
    context = device.make_context()
    
    # Código CUDA para la operación de cifrado
    aes_cuda_kernel = """
    __global__ void dummy_encrypt(unsigned char *data, int data_size) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < data_size) {
            // Operación de cifrado dummy (XOR con 0xFF)
            data[idx] = data[idx] ^ 0xff;
        }
    }
    """

    module = SourceModule(aes_cuda_kernel)
    dummy_encrypt = module.get_function("dummy_encrypt")

    # Adaptar la función decrypt_file para trabajar con datos en lugar de una ruta de archivo
    # Esto implica modificar la función para que acepte y procese datos en lugar de un archivo.
    decrypted_file_path= decrypt_file(file_data, dummy_encrypt)

    context.pop()
    context.detach()

    return decrypted_file_path


@app.route('/DESAES', methods=['POST'])
def AES_Desifrado():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Leemos los datos del archivo en memoria
        file_data = file.read()
        # Pasamos los datos directamente a la función AES
        decrypted_file_path = AESDES(file_data)

        

        return send_file(decrypted_file_path, as_attachment=True)
    return 'Something went wrong'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)










