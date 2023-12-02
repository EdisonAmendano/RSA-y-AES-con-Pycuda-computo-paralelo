import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { ServiceService } from '../../services/service.service';

@Component({
  selector: 'home',
  standalone: true,
  imports: [CommonModule, HttpClientModule],
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent {
  archivoSeleccionado!: File | null; // Inicialmente nulo
  nombreTXT: string = "";
  cargando: boolean = false; // Estado de carga

  constructor(private service: ServiceService) {}

  seleccionarArchivo(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.archivoSeleccionado = input.files[0];
    } else {
      this.archivoSeleccionado = null; // Resetear si no hay archivo seleccionado
    }
  }

  CodificarRSA() {
    this.iniciarCarga();
    this.nombreTXT = "CodificadoRSA";
    this.service.CodRSA(this.archivoSeleccionado!).subscribe(
      response => {
        this.finalizarCarga();
        this.descargarArchivo(response);
      },
      error => {
        this.finalizarCarga();
        console.log("no");
      }
    );
  }

  CodificarAES() {
    this.iniciarCarga();
    this.nombreTXT = "CodificadoAES";
    this.service.CodAES(this.archivoSeleccionado!).subscribe(
      response => {
        this.finalizarCarga();
        this.descargarArchivo(response);
      },
      error => {
        this.finalizarCarga();
        console.log("no");
      }
    );
  }

  DecodificarRSA() {
    this.iniciarCarga();
    this.nombreTXT = "DeCodificadoRSA";
    this.service.DeCodRSA(this.archivoSeleccionado!).subscribe(
      response => {
        this.finalizarCarga();
        this.descargarArchivo(response);
      },
      error => {
        this.finalizarCarga();
        console.log("no");
      }
    );
  }

  DecodificarAES() {
    this.iniciarCarga();
    this.nombreTXT = "DeCodificadoAES";
    this.service.DeCodAES(this.archivoSeleccionado!).subscribe(
      response => {
        this.finalizarCarga();
        this.descargarArchivo(response);
      },
      error => {
        this.finalizarCarga();
        console.log("no");
      }
    );
  }

  descargarArchivo(blob: Blob) {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = this.nombreTXT; // El nombre que quieres para el archivo descargado
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }

  private iniciarCarga() {
    this.cargando = true;
  }

  private finalizarCarga() {
    this.cargando = false;
  }
}
