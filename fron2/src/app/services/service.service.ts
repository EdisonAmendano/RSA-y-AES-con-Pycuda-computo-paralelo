import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class ServiceService {

  constructor(private http: HttpClient) { }
  
  CodRSA(archivo: File) {
    const formData: FormData = new FormData();
    formData.append('file', archivo, archivo.name);

    return this.http.post('http://localhost:5000/CIRSA', formData, {
      responseType: 'blob' // Si esperas un archivo como respuesta
    });
  }
  
  CodAES(archivo: File) {
    const formData: FormData = new FormData();
    formData.append('file', archivo, archivo.name);

    return this.http.post('http://localhost:5000/CiAES', formData, {
      responseType: 'blob' // Si esperas un archivo como respuesta
    });
  } 

  DeCodRSA(archivo: File) {
    const formData: FormData = new FormData();
    formData.append('file', archivo, archivo.name);

    return this.http.post('http://localhost:5000/DERSA', formData, {
      responseType: 'blob' // Si esperas un archivo como respuesta
    });
  } 

  DeCodAES(archivo: File) {
    const formData: FormData = new FormData();
    formData.append('file', archivo, archivo.name);

    return this.http.post('http://localhost:5000/DESAES', formData, {
      responseType: 'blob' // Si esperas un archivo como respuesta
    });
  } 
}
