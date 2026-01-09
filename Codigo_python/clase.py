#En cada paso voy a actualizar el libro con la informacion del tick, pasarselo de Input a la red neuronal, y obtener la prediccion
#Luego actualizo el libro para tener la informacion del siguiente tick
#Calculo la funcion de perdida entre la predicción de la red que vio el libro anterior y los valores del libro actual
import numpy as np
import math

class Libro:
    def __init__(self, activo):
        self.activo = activo

        #proxima fila a leer del csv
        self.indice = 0

        #datos para calcular features
        self.bid = np.zeros((5,2)) #5 precios de BID y 5 cantidades
        self.offer = np.zeros((5,2)) #5 precios de OFFER y 5 cantidades
        self.ultimo_trade = np.zeros(2) #precio, cantidad y tiempo desde el ultimo trade
        
        self.last_trade_time = 0
        self.last_time = 0
        self.time = 0 
        
        self.best_bid = 0
        self.last_best_bid = 0
        self.best_off = 0
        self.last_best_off = 0
        
        self.bid_volumen = 0
        self.off_volumen = 0
    
    def actualizar(self):
        side, level, price, size, time = self.__leer_siguiente_linea()
        self.__actualizar_tiempo(time, side)
        self.__actualizar_libro(side, level, price, size)
        self.indice += 1

        while (self.indice < len(self.activo) and self.activo.loc[self.indice, 'fecha_nano'] == self.time) or self.indice==0:
            side, level, price, size, time = self.__leer_siguiente_linea()
            self.__actualizar_tiempo(time, side)
            self.__actualizar_libro(side, level, price, size)
            self.indice += 1

        self.__reordenar_libro()

        if self.best_bid != self.bid[0][0] and self.bid[0][0] != 0:
            self.last_best_bid = self.best_bid
            self.best_bid = self.bid[0][0]

        if self.best_off != self.offer[0][0] and self.offer[0][0] != 0:
            self.last_best_off = self.best_off
            self.best_off = self.offer[0][0]

        self.bid_volumen = np.sum(self.bid[:][1])
        self.off_volumen = np.sum(self.offer[:][1])


    def obtener_features(self):
        time_since_last_trade = self.time - self.last_trade_time
        time_since_last_update = self.time - self.last_time
        mid_price =  (self.best_off + self.best_bid) / 2 
        spread = self.best_off - self.best_bid 
        volumen_bid = np.sum(self.bid[:,1])
        volumen_off = np.sum(self.offer[:,1])
        imbalance = ((volumen_bid - volumen_off) / (volumen_off + volumen_bid)) if (volumen_bid + volumen_off != 0) else 0
        vwap_bid = np.sum(self.bid[:,0] + self.bid[:,1]) / np.sum(self.bid[:,1]) if (np.sum(self.bid[:,1]) != 0) else 0
        vwap_offer = np.sum(self.offer[:,0] + self.offer[:,1]) / np.sum(self.offer[:,1]) if (np.sum(self.offer[:,1]) != 0) else 0
        relative_spread = spread / mid_price if mid_price != 0 else 0
        log_spread = math.log(self.best_off / self.best_bid) if self.best_bid != 0 else 0
        price_imbalance = (vwap_offer - vwap_bid) / mid_price if mid_price != 0 else 0
        depth_ratio = volumen_bid - volumen_off
        dirección_bid = 1 if self.best_bid > self.last_best_bid else 0
        dirección_offer = 1 if self.best_off > self.last_best_off else 0

        features = np.array([time_since_last_trade,
                             time_since_last_update,
                             mid_price,
                             spread,
                             volumen_bid,
                             volumen_off,
                             imbalance,
                             vwap_bid,
                             vwap_offer,
                             relative_spread,
                             log_spread,
                             price_imbalance,
                             depth_ratio, 
                             dirección_bid,
                             dirección_offer])
        return features

    def __leer_siguiente_linea(self):
        side = self.activo.loc[self.indice, 'side']
        level = self.activo.loc[self.indice, 'position']
        price = round(self.activo.loc[self.indice, 'price'], 6)
        size = self.activo.loc[self.indice, 'quantity']
        tiempo = self.activo.loc[self.indice, 'fecha_nano']
        return side, level, price, size, tiempo

    def __actualizar_libro(self, side, level, price, size):
        if (side == 'BI'):
            self.bid[level-1] = np.array([price, size])

        if (side == 'OF'):
            self.offer[level-1] = np.array([price, size])

        if (side == 'TRADE'):
            self.ultimo_trade = np.array([price, size])
            for i in range(5):
                if price == self.bid[i][0]:
                    self.bid[i][1] -= size
                    break
                elif price == self.offer[i][0]:
                    self.offer[i][1] -= size
                    break

    def __actualizar_tiempo(self, time, side):
        if time != self.time:
            self.last_time = self.time
            self.time = time

        if side == 'TRADE':
            self.last_trade_time = time

    def __reordenar_libro(self):
        unique_bid = {}
        unique_off = {}


        for k in range(5):
            price_bid = self.bid[k][0]
            vol_bid = self.bid[k][1]
            price_off = self.offer[k][0]
            vol_off = self.offer[k][1]


            if (price_bid > 0 and vol_bid > 0) and price_bid not in unique_bid:
                unique_bid[price_bid] = self.bid[k][1]

            if (price_off > 0 and vol_off > 0) and price_off not in unique_off:
                unique_off[price_off] = self.offer[k][1]

        sorted_bids = sorted(unique_bid.items(), key=lambda x: -x[0])  #de mayor a menor
        sorted_offs = sorted(unique_off.items(), key=lambda x: x[0])   #de menor a mayor

        for idx in range(5):
            if idx < len(sorted_bids):
                self.bid[idx] = np.array([sorted_bids[idx][0], sorted_bids[idx][1]])
            else:
                self.bid[idx] = np.array([0.0, 0.0])

            if idx < len(sorted_offs):
                self.offer[idx] = np.array([sorted_offs[idx][0], sorted_offs[idx][1]])
            else:
                self.offer[idx] = np.array([0.0, 0.0])
    
    def print(self):
        print("BID:")
        for i in range(5):
            print(f"Nivel {i+1}: Precio {self.bid[i][0]}, Cantidad {self.bid[i][1]}")
        print("OFFER:")
        for i in range(5):
            print(f"Nivel {i+1}: Precio {self.offer[i][0]}, Cantidad {self.offer[i][1]}")
        print(f"Ultimo Trade: Precio {self.ultimo_trade[0]}, Cantidad {self.ultimo_trade[1]}, tiempo {self.last_trade_time}")
        print(f"last_time: {self.last_time}")
        print(f"time: {self.time}")
        print(f"best_bid: {self.best_bid}, last_best_bid {self.last_best_bid}")
        print(f"best_off: {self.best_off}, last_best_off {self.last_best_off}")