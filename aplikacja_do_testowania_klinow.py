"""
    Aplikacja do sprawdzania stanu zaklinowania w generatorze na podstawie odebranego sygnału z akcelometru.
    Aplikacja umożliwia wprowadzenie liczby żłobków(rzędów), ilości klinów oraz całkowitej długości każdego rzędu.
    Te informacje przesyłane są do kontrolera, który rozpoczyna pomiary, przesyłane do komputera z wykorzystaniem portu szeregowego.
    Otrzymane dane są zapisywane w bazach danych. Z zebranych pomiarów wyliczane jest środkowe pasmo widmowe (inspo reserachergate todo analiza KNN, SVN).
    Dane są wizualizowane w dziedzinie czasu oraz częstotliości dla danego zbioru (rząd, klin). Do excela można eksportować tabelę z elementami: rząd, klin, spectral centroid.
"""

import sys
import time
import serial
import sqlite3
import datetime
import csv
from dataclasses import dataclass
import numpy as np

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QFormLayout, QPushButton, QLineEdit, QMessageBox, QListWidget, QVBoxLayout, QHBoxLayout, 
    QTableWidget, QTableWidgetItem, QFileDialog, QHeaderView, QLabel)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtSerialPort import QSerialPortInfo
import pyqtgraph as pg


@dataclass
class Packet:
    row: int
    wedge: int 
    samples: list[int]


def calculate_spectral_centroid(samples: list[int], samplerate: int = 100) -> float:
    """Oblicza środkowe pasmo widmowe (Spectral Centroid) w Hz."""
    if not samples or len(samples) == 0:
        return 0.0
    x = np.array(samples)
    magnitudes = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / samplerate)
    sum_mag = np.sum(magnitudes)
    if sum_mag == 0:
        return 0.0
    return float(np.sum(magnitudes * freqs) / sum_mag)


""""
Okno startowe, gdzie użytkownik wprowadza zmienne takie jak: ilość rzędów, ilosć klinów oraz całkowitą długość pojedynczego rzędu.
Wybierany jest również port szeregowy do którego podpięty jest mikrokontroler
Parametrami domyślnymi są: czas trwania pomiarów po uderzeniu młotkiem (duration) oraz częstotliowść próbkowania (sampling_rate) 
"""
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        print('Aplikacja uruchomiona')
        self.setWindowTitle("Aplikacja do testowania stanu zaklinowania")
        self.controller = None

        # Pola tekstowe
        self.ilosc_rzedow = QLineEdit()
        self.ilosc_klinow = QLineEdit()
        self.calkowita_dlugosc = QLineEdit()

        # Wybór portu COM
        self.list_widget = QListWidget(self)
        self.refresh_ports()

        # Przycisk odświeżania portów
        self.btn_refresh = QPushButton("Odśwież listę portów COM")
        self.btn_refresh.clicked.connect(self.refresh_ports)

        # Przycisk startowy
        self.button = QPushButton("Zatwierdź dane i rozpocznij pomiary")
        self.button.clicked.connect(self.start_measurement)

        # Layout
        layout = QFormLayout()
        layout.addRow("Liczba rzędów:", self.ilosc_rzedow)
        layout.addRow("Liczba klinów w rzędzie:", self.ilosc_klinow)
        layout.addRow("Całkowita długość klinów [mm]:", self.calkowita_dlugosc)
        layout.addRow("Wybierz port COM:", self.list_widget)
        layout.addRow(self.btn_refresh)
        layout.addRow(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def refresh_ports(self):
        """Odświeża listę dostępnych portów szeregowych."""
        self.list_widget.clear()
        ports = QSerialPortInfo.availablePorts()
        for port in ports:
            self.list_widget.addItem(port.portName())

    def start_measurement(self):
        """Sprawdzenie danych i rozpoczęcie pomiarów."""
        item = self.list_widget.currentItem()
        if item is None:
            QMessageBox.warning(self, "Błąd", "Wybierz port COM z listy.")
            return
        port = item.text()
        try:
            rows = int(self.ilosc_rzedow.text())
            wedges = int(self.ilosc_klinow.text())
            length = int(self.calkowita_dlugosc.text())
        except ValueError:
            QMessageBox.warning(self, "Błąd", "Wprowadź poprawne wartości liczbowe.")
            return

        print(f"Start: Port={port}, Rzędy={rows}, Kliny={wedges}, Długość={length}")

        # Inicjalizacja kontrolera pomiarów
        self.controller = MeasurementController(port, rows, wedges, length)


class MeasurementController:
    def __init__(self, portname, rows, wedges, length):
        self.sample_rate = 100
        self.duration = 2
        self.rows = rows
        self.wedges = wedges
        self.length = length

        self.db = DataBaseManager()
        self.experiment_id = self.db.create_experiment(
            rows=rows, wedges=wedges, length=length, 
            sampling_rate=self.sample_rate, duration=self.duration
        )

        self.serial = SerialManager(portname=portname, sample_rate=self.sample_rate, duration=self.duration)
        self.okno_wykresow = PlotWindow(sample_rate=self.sample_rate, duration=self.duration)
        self.okno_wykresow.show()

        # Połączenie sygnałów z wątku szeregowego
        self.serial.data_received.connect(self.on_packet_received)
        self.serial.finished_signal.connect(self.on_measurement_finished)

        # Uruchamiamy wątek nasłuchujący
        self.serial.start()

        # Wysyłanie konfiguracji do Arduino dopiero gdy port jest w pełni gotowy
        config_msg = f"{rows},{wedges},{length},{self.sample_rate},{self.duration}\n"
        print(f"Wysyłanie konfiguracji do Arduino: {config_msg.strip()}")
        self.serial.send_data(config_msg)

    def on_packet_received(self, packet: Packet):
        """Obsługa pojedynczej paczki próbek z Arduino."""
        measurement_id = self.db.create_measurement(self.experiment_id, packet.row, packet.wedge)
        self.db.insert_sample(measurement_id, packet.samples)
        self.okno_wykresow.plot_graphs(data=packet.samples)
        print(f"Odebrano pakiet [Row {packet.row}, Wedge {packet.wedge}], Liczba próbek: {len(packet.samples)}")

    def on_measurement_finished(self):
        """Obsługa zakończenia wszystkich pomiarów."""
        print("Pomiary zakończone. Pobieranie danych i wyliczanie wskaźników...")
        
        measurements_data = self.db.get_experiment_data(self.experiment_id)

        processed_results = []
        for m in measurements_data:
            sc_value = calculate_spectral_centroid(m['samples'], self.sample_rate)
            processed_results.append({
                'row': m['row'],
                'wedge': m['wedge'],
                'sample_count': len(m['samples']),
                'spectral_centroid': sc_value
            })

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_csv_filename = f"wyniki_zaklinowania_{timestamp}.csv"
        self.save_to_csv(auto_csv_filename, processed_results)

        self.okno_wykresow.close()
        self.okno_wynikow = DataVisualizeWindow(
            processed_results=processed_results, 
            rows=self.rows, 
            wedges=self.wedges,
            auto_csv_path=auto_csv_filename
        )
        self.okno_wynikow.show()

    def save_to_csv(self, filename: str, results: list[dict]):
        """Zapisuje wyniki do pliku CSV."""
        try:
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(['Rząd (Row)', 'Klin (Wedge)', 'Liczba próbek', 'Spectral Centroid [Hz]'])
                for item in results:
                    writer.writerow([
                        item['row'], 
                        item['wedge'], 
                        item['sample_count'], 
                        f"{item['spectral_centroid']:.2f}"
                    ])
            print(f"Dane pomyślnie zapisane do pliku: {filename}")
        except Exception as e:
            print(f"Błąd zapisu CSV: {e}")


class SerialManager(QThread):
    data_received = pyqtSignal(Packet)
    finished_signal = pyqtSignal()

    def __init__(self, portname="COM3", sample_rate=100, duration=5):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.ser = None
        try:
            # Prędkość dostosowana do Arduino: 115200 baud
            self.ser = serial.Serial(portname, 115200, timeout=1)
            
            # Oczekiwanie 2 sekundy na uruchomienie bootloadera po resecie DTR
            time.sleep(2.0)
            
            # Czyszczenie bufora wejściowego/wyjściowego po starcie
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            print(f"Połączono z portem {self.ser.name}")
        except Exception as e:
            print(f"Błąd portu szeregowego: {e}")

    def send_data(self, msg):
        if self.ser and self.ser.is_open:
            self.ser.write(msg.encode('utf-8'))
            self.ser.flush()  # Wymuszenie natychmiastowego wysłania z bufora systemowego

    def receive_measurement(self):
        if not self.ser or not self.ser.is_open:
            return None

        try:
            line = self.ser.readline().decode(errors='ignore').strip()
        except Exception as e:
            print(f"Błąd odczytu z portu: {e}")
            return None

        if not line:
            return None

        if line == "Koniec pomiarow":
            print('Odebrano sygnał: Koniec pomiarów')
            self.finished_signal.emit()
            return "FINISHED"

        if line == "OTRZYMANO PARAM, PRZESYL DANYCH":
            print('Parametry zostały pomyślnie odczytane przez Arduino')
            return "OTRZYMANO PARAM"

        if line.startswith("ERROR:"):
            print(f"Błąd zgłoszony przez Arduino: {line}")
            return None

        if "," not in line:
            return None

        # Nagłówek pakietu "row,wedge"
        pkg = Packet(0, 0, [])
        try:
            pkg.row, pkg.wedge = map(int, line.split(","))
        except ValueError:
            return None

        # Odczyt próbek aż do napisu "END"
        while True:
            try:
                sample_line = self.ser.readline().decode(errors='ignore').strip()
            except Exception:
                break

            if sample_line == "END":
                break
            if not sample_line:
                continue
            try:
                pkg.samples.append(int(sample_line))
            except ValueError:
                continue

        return pkg

    def run(self):
        """Pętla wątku odbierającego dane z Arduino."""
        if not self.ser or not self.ser.is_open:
            return

        while self.ser.is_open:
            packet = self.receive_measurement()
            if packet == "FINISHED":
                break
            elif isinstance(packet, Packet) and packet.samples:
                self.data_received.emit(packet)


class DataBaseManager:
    def __init__(self):
        db_name = f"experiment_{datetime.date.today()}.db"
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA foreign_keys = ON")
        self.create_tables()

    def create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS Experiment(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                rows INTEGER,
                wedges INTEGER,
                length REAL,
                sampling_rate INTEGER,
                duration REAL)""")
   
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS Measurement(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                row_number INTEGER,
                wedge_number INTEGER,
                FOREIGN KEY(experiment_id) REFERENCES Experiment(id))""")

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS Sample(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                measurement_id INTEGER,
                sample_index INTEGER,
                adc_value INTEGER,
                FOREIGN KEY(measurement_id) REFERENCES Measurement(id))""")
        
        self.conn.commit()

    def create_experiment(self, rows, wedges, length, sampling_rate, duration):
        self.cursor.execute("""
            INSERT INTO Experiment(date, rows, wedges, length, sampling_rate, duration)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (str(datetime.datetime.now()), rows, wedges, length, sampling_rate, duration))
        self.conn.commit()
        return self.cursor.lastrowid

    def create_measurement(self, experiment_id, row_number, wedge_number):
        self.cursor.execute("""
            INSERT INTO Measurement(experiment_id, row_number, wedge_number)
            VALUES (?, ?, ?)""", (experiment_id, row_number, wedge_number))
        self.conn.commit()
        return self.cursor.lastrowid

    def insert_sample(self, measurement_id, adc_values):
        records = [(measurement_id, i, val) for i, val in enumerate(adc_values)]
        self.cursor.executemany("""
            INSERT INTO Sample(measurement_id, sample_index, adc_value)
            VALUES (?, ?, ?)""", records)
        self.conn.commit()

    def get_experiment_data(self, experiment_id):
        self.cursor.execute("""
            SELECT id, row_number, wedge_number 
            FROM Measurement 
            WHERE experiment_id = ?
            ORDER BY row_number, wedge_number""", (experiment_id,))
        measurements = self.cursor.fetchall()

        results = []
        for m_id, row, wedge in measurements:
            self.cursor.execute("""
                SELECT adc_value FROM Sample 
                WHERE measurement_id = ? 
                ORDER BY sample_index""", (m_id,))
            samples = [r[0] for r in self.cursor.fetchall()]
            results.append({
                'row': row,
                'wedge': wedge,
                'samples': samples
            })
        return results


class PlotWindow(QMainWindow):
    def __init__(self, sample_rate, duration):
        super().__init__()
        self.setWindowTitle("Podgląd bieżącego pomiaru")
        self.resize(800, 600)

        self.sample_rate = sample_rate
        self.duration = duration

        central_widget = QWidget()
        layout = QVBoxLayout()

        self.plot_graph_time = pg.PlotWidget()
        self.plot_graph_time.setBackground("w")
        self.plot_graph_time.setYRange(0, 1000, padding=0)
        
        self.plot_graph_fft = pg.PlotWidget()
        self.plot_graph_fft.setBackground("w")

        layout.addWidget(self.plot_graph_time)
        layout.addWidget(self.plot_graph_fft)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def plot_graphs(self, data):
        self.plot_graph_time.clear()
        self.plot_graph_fft.clear()

        time = np.linspace(0, self.duration, len(data), endpoint=False)
        self.plot_graph_time.plot(time, data, pen='b')
        self.plot_graph_time.setTitle("Sygnał czasowy")
        self.plot_graph_time.setLabel("left", "Amplituda")
        self.plot_graph_time.setLabel("bottom", "Czas [s]")

        freq = np.fft.rfftfreq(len(data), d=1/self.sample_rate)
        fft_values = np.abs(np.fft.rfft(data))

        self.plot_graph_fft.plot(freq, fft_values, pen='r')
        self.plot_graph_fft.setTitle("Widmo częstotliwościowe (FFT)")
        self.plot_graph_fft.setLabel("left", "Amplituda")
        self.plot_graph_fft.setLabel("bottom", "Częstotliwość [Hz]")


class DataVisualizeWindow(QMainWindow):
    def __init__(self, processed_results: list[dict], rows: int, wedges: int, auto_csv_path: str):
        super().__init__()
        self.setWindowTitle('Podsumowanie wyników zaklinowania')
        self.resize(650, 450)
        self.results = processed_results

        layout = QVBoxLayout()

        label_info = QLabel(f"<b>Pomiary zakończone.</b> Łącznie rzędów: {rows}, Klinów w rzędzie: {wedges}<br>"
                            f"<i>Automatyczny zapis: {auto_csv_path}</i>")
        layout.addWidget(label_info)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['Rząd (Row)', 'Klin (Wedge)', 'Liczba próbek', 'Spectral Centroid [Hz]'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.table.setRowCount(len(processed_results))
        for row_idx, item in enumerate(processed_results):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(item['row'])))
            self.table.setItem(row_idx, 1, QTableWidgetItem(str(item['wedge'])))
            self.table.setItem(row_idx, 2, QTableWidgetItem(str(item['sample_count'])))
            self.table.setItem(row_idx, 3, QTableWidgetItem(f"{item['spectral_centroid']:.2f}"))

        layout.addWidget(self.table)

        self.btn_export = QPushButton("Eksportuj dane do pliku CSV...")
        self.btn_export.clicked.connect(self.export_csv_dialog)
        layout.addWidget(self.btn_export)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def export_csv_dialog(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Zapisz wyniki do CSV", "", "Pliki CSV (*.csv)")
        if filepath:
            try:
                with open(filepath, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file, delimiter=';')
                    writer.writerow(['Rząd (Row)', 'Klin (Wedge)', 'Liczba próbek', 'Spectral Centroid [Hz]'])
                    for item in self.results:
                        writer.writerow([
                            item['row'], 
                            item['wedge'], 
                            item['sample_count'], 
                            f"{item['spectral_centroid']:.2f}"
                        ])
                QMessageBox.information(self, "Sukces", f"Zapisano dane do:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Błąd", f"Nie udało się zapisać pliku: {e}")


def main():        
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()