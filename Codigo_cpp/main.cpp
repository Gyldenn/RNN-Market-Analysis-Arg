#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <fstream>
#include <sstream>
#include <array>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>


class Libro {
public:
    Libro(const std::string& csv_path)
        : indice(0), time(0), last_time(0), last_trade_time(0),
          best_bid(0), last_best_bid(0), best_off(0), last_best_off(0),
          bid_volumen(0), off_volumen(0)
    {
		bid.fill({0.0f, 0.0f});
        offer.fill({0.0f, 0.0f});
        ultimo_trade = {0.0f, 0.0f};
        cargar_csv(csv_path);
    }

    void actualizar() {
        if (indice >= activo.size()) return;

        int side, level;
        float price, size;
        int64_t timestamp;

        leer_siguiente_linea(side, level, price, size, timestamp);
        actualizar_tiempo(timestamp, side);
        actualizar_libro(side, level, price, size);
        ++indice;

        // Leer todas las filas con el mismo timestamp
        while (indice+1 < activo.size() && activo[indice][4] == time) {
            leer_siguiente_linea(side, level, price, size, timestamp);
            actualizar_tiempo(timestamp, side);
            actualizar_libro(side, level, price, size);
            ++indice;
        }

        reordenar_libro();

        if (best_bid != bid[0][0] && bid[0][0] != 0) {
            last_best_bid = best_bid;
            best_bid = bid[0][0];
        }
        if (best_off != offer[0][0] && offer[0][0] != 0) {
            last_best_off = best_off;
            best_off = offer[0][0];
        }

        bid_volumen = 0.0f;
        off_volumen = 0.0f;
        for (int i = 0; i < 5; ++i) {
            bid_volumen += bid[i][1];
            off_volumen += offer[i][1];
        }
    }

    torch::Tensor obtener_features() const {
        best_off == (best_off != 0) ? best_off : last_best_off;
        best_bid == (best_bid != 0) ? best_bid : last_best_bid;
        float time_since_last_trade = static_cast<float>(time - last_trade_time);
        float time_since_last_update = static_cast<float>(time - last_time);
        float mid_price = (best_off + best_bid) / 2.0f;
        float spread = best_off - best_bid;
        float imbalance = (bid_volumen + off_volumen != 0) ? (bid_volumen - off_volumen) / (bid_volumen + off_volumen) : 0.0f;

        float vwap_bid = 0.0f, vwap_offer = 0.0f;
        float sum_bid_vol = 0.0f, sum_off_vol = 0.0f;
        for (int i = 0; i < 5; ++i) {
            vwap_bid += bid[i][0] * bid[i][1];
            sum_bid_vol += bid[i][1];
            vwap_offer += offer[i][0] * offer[i][1];
            sum_off_vol += offer[i][1];
        }
        if (sum_bid_vol != 0) vwap_bid /= sum_bid_vol;
        if (sum_off_vol != 0) vwap_offer /= sum_off_vol;

        float relative_spread = (mid_price != 0) ? spread / mid_price : 0.0f;
        float log_spread = (best_bid != 0) ? std::log(best_off / best_bid) : 0.0f;
        float price_imbalance = (mid_price != 0) ? (vwap_offer - vwap_bid) / mid_price : 0.0f;
        float depth_ratio = bid_volumen - off_volumen;
        float direccion_bid = (best_bid > last_best_bid) ? 1.0f : 0.0f;
        float direccion_offer = (best_off > last_best_off) ? 1.0f : 0.0f;

        std::array<float,15> features = {
            time_since_last_trade,
            time_since_last_update,
            mid_price,
            spread,
            bid_volumen,
            off_volumen,
            imbalance,
            vwap_bid,
            vwap_offer,
            relative_spread,
            log_spread,
            price_imbalance,
            depth_ratio,
            direccion_bid,
            direccion_offer
        };

        return torch::from_blob((void*)features.data(), {15}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    }

    size_t size() const { return activo.size(); }
    size_t indice_actual() const { return indice; }

	bool tiene_siguiente() const { return indice < activo.size()-1; }

private:
    // Datos del CSV: cada fila es {side, level, price, size, timestamp}
    std::vector<std::array<float,5>> activo;

    // Estado interno
    std::array<std::array<float,2>,5> bid;
    std::array<std::array<float,2>,5> offer;
    std::array<float,2> ultimo_trade;

    int64_t indice;
    int64_t time, last_time, last_trade_time;
    float best_bid, last_best_bid, best_off, last_best_off;
    float bid_volumen, off_volumen;

    // Side codificado: BI=0, OF=1, TRADE=2
    void leer_siguiente_linea(int &side, int &level, float &price, float &size, int64_t &timestamp) {
        auto& row = activo[indice];
        side = static_cast<int>(row[0]);
        level = static_cast<int>(row[1]);
        price = row[2];
        size = row[3];
        timestamp = static_cast<int64_t>(row[4]);
    }

    void actualizar_libro(int side, int level, float price, float size) {
        if (side == 0) bid[level-1] = {price, size};
        else if (side == 1) offer[level-1] = {price, size};
        else if (side == 2) {
            ultimo_trade = {price, size};
            for (int i = 0; i < 5; ++i) {
                if (bid[i][0] == price) { bid[i][1] -= size; break; }
                else if (offer[i][0] == price) { offer[i][1] -= size; break; }
            }
        }
    }

    void actualizar_tiempo(int64_t t, int side) {
        if (t != time) { last_time = time; time = t; }
        if (side == 2) last_trade_time = t;
    }

    void reordenar_libro() {
        std::vector<std::pair<float,float>> unique_bid, unique_offer;
        for (int i = 0; i < 5; ++i) {
            if (bid[i][0] > 0 && bid[i][1] > 0) unique_bid.emplace_back(bid[i][0], bid[i][1]);
            if (offer[i][0] > 0 && offer[i][1] > 0) unique_offer.emplace_back(offer[i][0], offer[i][1]);
        }

        std::sort(unique_bid.begin(), unique_bid.end(), [](auto &a, auto &b){ return a.first > b.first; });
        std::sort(unique_offer.begin(), unique_offer.end(), [](auto &a, auto &b){ return a.first < b.first; });

        for (int i = 0; i < 5; ++i) {
            bid[i] = (i < unique_bid.size()) ? std::array<float,2>{unique_bid[i].first, unique_bid[i].second} : std::array<float,2>{0.0f,0.0f};
            offer[i] = (i < unique_offer.size()) ? std::array<float,2>{unique_offer[i].first, unique_offer[i].second} : std::array<float,2>{0.0f,0.0f};
        }
    }

    void cargar_csv(const std::string& path) {
            
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("No se pudo abrir el CSV");

        std::fseek(f, 0, SEEK_END);
        size_t file_size = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);

        std::vector<char> buffer(file_size + 1);
        size_t read_bytes = std::fread(buffer.data(), 1, file_size, f);
        std::fclose(f);
        buffer[read_bytes] = '\0';

        char* ptr = buffer.data();
        char* end = buffer.data() + read_bytes;

        while (ptr < end) {
            std::array<float,5> row;

            // Saltar las primeras 3 columnas
            for (int i = 0; i < 3; ++i) {
                while (ptr < end && *ptr != ',') ++ptr;
                if (ptr < end && *ptr == ',') ++ptr;
            }

            // side
            if (std::strncmp(ptr, "BI", 2) == 0) row[0] = 0, ptr += 3;
            else if(std::strncmp(ptr, "OF", 2) == 0) row[0] = 1, ptr += 3;
            else row[0] = 2; // TRADE

            // level
            row[1] = static_cast<float>(std::strtol(ptr, &ptr, 10));
            if (*ptr == ',') ++ptr;

            // price
            row[2] = std::strtof(ptr, &ptr);
            if (*ptr == ',') ++ptr;

            // size
            row[3] = std::strtof(ptr, &ptr);
            if (*ptr == ',') ++ptr;

            // timestamp
            row[4] = static_cast<float>(std::strtoll(ptr, &ptr, 10));

            // saltar newline
            while (ptr < end && (*ptr == '\n' || *ptr == '\r')) ++ptr;

            activo.push_back(row);
        }
    }
};


int main() {
    // --- Rutas fijas ---
    const std::string modelo_path = "../Modelo_entrenado.pt";
    const std::string csv_path = "../datos.csv";

    try {
        // --- Cargar modelo TorchScript ---
        torch::jit::script::Module model = torch::jit::load(modelo_path);
        model.eval();
        std::cout << "Modelo cargado correctamente.\n";

        // --- Cargar CSV y crear objeto Libro ---
        Libro libro(csv_path);
        std::cout << "Datos cargados: " << libro.size() << " filas.\n"; //esto lo corre sin problema

        std::cout << libro.indice_actual(); //esto no aparece, pero no hay nada en el medio antes del error

        torch::Tensor ultima_prediccion;


        // --- Iterar sobre el dataset ---
        while (libro.tiene_siguiente()) {

            libro.actualizar();
            torch::Tensor features = libro.obtener_features();
            features = features.unsqueeze(0); // [1, input_dim] → batch de 1

            std::vector<torch::jit::IValue> inputs;
            if (!features.defined()) {
                std::cerr << "Features no definidos, salto iteración\n";
                continue;
            }
            inputs.push_back(features);

            torch::Tensor pred = model.forward(inputs).toTensor();
            ultima_prediccion = pred;

            // Opcional: imprimir evolución
            std::cout << pred << std::endl;
        }

        // --- Mostrar predicción final ---
        if (ultima_prediccion.defined()) {
            std::cout << "Predicción final:\n" << ultima_prediccion << "\n";
        } else {
            std::cerr << "No se generó ninguna predicción (CSV vacío?).\n";
        }

    } catch (const c10::Error& e) {
        std::cerr << "Error cargando o ejecutando el modelo: " << e.what() << "\n";
        return -1;
    }

    return 0;
}