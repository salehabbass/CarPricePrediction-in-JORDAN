import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib
import sklearn


enc_1 = joblib.load("encoder_1.pkl")
enc_2 = joblib.load("encoder_2.pkl")
enc_3 = joblib.load("encoder_3.pkl")
enc_4 = joblib.load("encoder_4.pkl")
enc_5 = joblib.load("encoder_5.pkl")
enc_6 = joblib.load("encoder_6.pkl")

loaded_scaler = joblib.load("MinMaxScaler.pkl")
loaded_model = joblib.load("RandomForestRegressor.pkl")


def predict_price(features):
    pred = loaded_model.predict(features)
    return pred

def predict():
    # Retrieve feature values
    make = make_var.get()
    model = model_var.get()
    year = int(year_var.get())
    trans = trans_var.get()
    kilometers = kilo_var.get()
    fuel = fuel_var.get()
    engine_size = engine_size_var.get()
    
    
    make_value = enc_1.transform(np.array([make]))
    model_value = enc_2.transform(np.array([model]))
    year_value = loaded_scaler.transform(np.array([[year]]))
    trans_value = enc_3.transform(np.array([trans]))
    kilo_value = enc_4.transform(np.array([kilometers]))
    fuel_value = enc_5.transform(np.array([fuel]))
    engine_value = enc_6.transform(np.array([engine_size]))

    
    # Create feature vector
    X = np.array([make_value, model_value, year_value,trans_value, kilo_value, fuel_value, engine_value]).reshape(1, -1)
    


    # Predict car price
    predicted_price = predict_price(X)
    
    # Update result label
    
    result_label.config(text=f"Predicted Price: {int(predicted_price[0])} JD")

def clear():
    make_combobox.delete(0,tk.END)
    model_combobox.delete(0,tk.END)
    trans_combobox.delete(0,tk.END)
    fuel_combobox.delete(0,tk.END)
    engine_size_entry.delete(0,tk.END)
    kilometers_combobox.delete(0,tk.END)
    year_combobox.delete(0,tk.END)
    result_label.config(text="")

# Create Tkinter window
window = tk.Tk()
window.title("Car Price Predictor")
window.iconbitmap('icon.ico')


# Create labels
label = tk.Label(window, text="predict your car price by insert the following features", font=("Arial", 14))
label.grid(row=0, column=0, padx=0, pady=0, sticky="w")

make_label = ttk.Label(window, text="Car Make:")
make_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

model_label = ttk.Label(window, text="Car Model:")
model_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

year_label = ttk.Label(window, text = "year:")
year_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")

trans_label = ttk.Label(window, text = "transmision:")
trans_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")

kilometers_label = ttk.Label(window, text="Kilometers:")
kilometers_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")

fuel_label = ttk.Label(window, text="Fuel:")
fuel_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")

engine_size_label = ttk.Label(window, text="Engine Size:")
engine_size_label.grid(row=7, column=0, padx=5, pady=5, sticky="w")

result_label = ttk.Label(window, text="",font=("Arial", 13))
result_label.grid(row=8, column=0, padx=5, pady=5, sticky="w")

# Create entry widgets and dropdowns
make_var = tk.StringVar()
make_combobox = ttk.Combobox(window, textvariable=make_var, values=['Toyota', 'Mercedes Benz', 'Lexus', 'BMW', 'Mitsubishi', 'Hyundai',
       'Honda', 'Tesla', 'Ford', 'Audi', 'Kia', 'Land Rover', 'Chevrolet',
       'SEAT', 'Nissan', 'Changan', 'Volkswagen', 'Mazda', 'Infiniti',
       'Isuzu', 'Porsche', 'Lincoln', 'Peugeot', 'GMC', 'Jeep', 'BYD',
       'Saab', 'Dodge', 'Opel', 'Dongfeng', 'Great Wall', 'Haval', 'Fiat',
       'Cadillac', 'Suzuki', 'MG', 'Daihatsu', 'Renault', 'Citroen',
       'BAIC', 'Samsung', 'Daewoo', 'Smart', 'Mahindra', 'Polestar',
       'Hummer', 'Chrysler', 'Geely', 'Lada', 'Chery', 'Subaru',
       'Hunaghai', 'MINI', 'Proton', 'Jac', 'Jaguar', 'Aston Martin',
       'JMC', 'Pontiac', 'Skoda', 'Leapmotor', 'Skywell', 'Bestune',
       'GAC'])
make_combobox.grid(row=1, column=1, padx=5, pady=5)

model_var = tk.StringVar()
model_combobox = ttk.Combobox(window, textvariable=model_var, values=['Prius', 'C 180', 'CT 200', '530', 'Pajero', 'Prius C', 'Kona',
       'E 200', 'e:NS1', 'Sonata', '3', 'Lancer', 'Fusion', 'Q8 e-tron',
       'Civic', 'Ioniq', 'Niro', 'LR4', 'Ioniq 5', 'Bolt', 'Sportage',
       'Leone', 'F-150', 'Insight', 'Leaf', 'E-Star', 'Morning', 'Porter',
       'Optima K5', 'ID.6', 'Vito', 'Avante', 'C 350e', 'QX80', 'E 350',
       'Accord', 'Verna', 'Menlo', 'bZ4X', 'E-Golf', 'X6', 'ID.4',
       'C 200 Coupe', 'Camry', 'NPR', 'D-Max', 'Jetta', 'Pegas', 'Macan',
       'K3', 'Genesis Coupe', 'MKZ', 'Corolla', '530e', 'Accent', 'e:NP1',
       'RAV 4', 'Y', 'L200', '325', 'Mustang', 'i20', 'Forte', 'C-MAX',
       'CLA 45 AMG', '308', 'ES', 'Yukon', 'Gladiator', 'Bongo',
       'Veloster', 'Focus', 'Sylphy', 'Highlander', 'Yaris', 'E2', '330',
       'C 250', 'X', 'Tucson', 'E-Bora', '95', 'EV6', 'Picanto', 'Acadia',
       'Spectra', 'E-lavida', 'Terrain', 'i3', 'Rogue', 'Malibu',
       'Tiguan', 'Ram', 'CX30', 'Captiva', 'Liberty', 'Q7', 'i10',
       'Vectra', 'CLK 200', 'S', 'Compass', 'X5', 'Camaro', 'A30 Sport',
       'Juke', 'Caddy', 'Han EV', 'Cerato', 'Grand Cherokee', 'Optima',
       'EQA', 'Seagull', 'Cobalt', 'Outlander', 'Explorer', '206',
       'ES 300', 'Yuan', 'CLK 320', 'Golf MK4', 'Sonic', 'H-1 Starex',
       'Pathfinder', 'Tundra', 'Sunny', 'Wingle', 'Rio', 'NKR', 'Hilux',
       'Eado', 'Jolion', '500e', 'C 220', '523', 'Han', 'S 350', 'GS 460',
       'C 250 Coupe', 'Sierra', 'A6', 'Escape', 'Touareg', 'Escalade',
       'Poer', 'Land Cruiser', 'Range Rover Vogue', 'Swift', 'CLS 350',
       'ML 350', 'C 200', 'MG HS', 'Range Rover Sport', 'Q5', 'Boxster S',
       'Sirion', 'Duster', 'Prado', 'Polo', 'Elantra', 'LX 570',
       'Colorado', 'Mirage', 'C-HR', 'Spark', 'SE 300', 'Elysée',
       'Song Plus', 'IS 300', 'E 250', 'CX-9', 'H6 Hev', 'ID.3', 'CR-V',
       'D20', 'bZ3X', 'Fluence', 'SL 500', 'Mighty', 'MG ZS EV',
       'Celerio', 'Galant', 'Getz', 'RX 450', 'LS 460', 'NX 300', '740',
       'HS', 'Navara', 'C3', 'Passat', 'S 280', 'Tahoe', 'Silverado',
       'Alto', 'Golf', '730', 'E 300', 'Koleos', 'G 55 AMG', '520',
       'Azera', 'S 400', 'E 250 Coupe', 'Traverse', 'Corolla Cross',
       'Grand Vitara', 'SM 3', 'Lotze', 'Atos', '307', 'Altima', 'Omega',
       'Tiida', 'Canter', 'Santa Fe', 'Shuma', 'E 190', 'Sephia', 'FSR',
       'MDX', 'Astra', 'Nubira', 'Golf MK3', 'H100', 'Kadett', 'MG 5',
       'Logan', 'GTI', 'C4', 'Partner', 'Durango', 'Challenger', 'Aveo',
       'Mira', 'V3', 'Blazer', 'ASX', '207', 'Forspeed', 'QX60', 'Prius+',
       'Transporter', 'X-Trail', 'Escort', 'Caprice', 'Murano', 'Pickup',
       'Cayman S', 'Kicks', 'Sentra', 'Fortuner', 'BLS', 'ALSVIN',
       'Avanza', '407', 'Creta', 'Lanos', 'Optra', 'Scorpio', 'Fortow',
       'Genesis', 'KB', 'EQC', '2008', 'Clio', 'GLC 300', 'Pride',
       'Polestar 2', '525', 'Trajet', 'Cayenne', 'ES 350', 'Samurai',
       'GLC 200', 'E 240', 'QM 5', 'CTS', 'Tercel', 'Megane', 'Cruze',
       '2', 'City', 'Fuso Canter', '4007', 'Soul', 'Cs15', 'Ranger', 'H6',
       'Canyon', 'E46', 'RX', 'Fuso Rosa', 'Carnival', 'Micra',
       'GLC 350e Coupe', 'Vita', 'Delta', 'Q3', 'Frontier', 'Avalon',
       'H3', 'Trax', 'Santamo', '300C', 'Pregio', 'Dolphin', 'e-tron',
       'Excel', 'A 200', '5008', '208', 'Sandero', '318', 'CLA 200',
       'B 250', 'X6M', '107', 'Qin', 'Sorento', 'Fuso', 'C2', 'Echo',
       'E 350e', '316', '301', 'Passat CC', 'Boxer', 'CS35 PLUS',
       'Freelander', 'Sprinter', 'Insignia', 'Lumin', 'Zoe', '3000GT',
       'Epica', 'Cadenza', 'Envoy', 'Cressida', 'GS', 'Berlingo',
       'G 500 4x4²', 'Lacetti', 'SRX', 'Jazz', 'Z3', 'Cherokee', 'Zafira',
       'Corona', 'S 320', '320', 'Beetle', 'E30', 'Nitro', 'Tiburon',
       'A4', 'Charger', 'Bora', 'Vito 116', 'Azkarra', 'Sonet', 'Dyna',
       'DS6', '4x4 Urban', 'C 230', 'Versa', 'Cielo', 'EcoSport', 'UX',
       'Eado Plus', 'TrailBlazer', 'Attrage', '323', 'EON', 'Volt',
       'Hiace', 'Tiggo', 'Junior', 'Legacy', 'X3', 'MG GT', 'SX4',
       '4Runner', 'E 320', 'DD6534GL', 'G35', '6', 'Legnum', 'Forfour',
       'Tacoma', '500', 'Neon', 'Citan', 'M3', 'Sedona', 'Q2', 'Tuscani',
       '2000GT', 'Grandis', 'Cooper', 'Waja', 'Vito 111', 'Cayenne GTS',
       'SL 450', '5', 'Vito 114', 'HR-V', 'Taurus', 'FCX Clarity', 'XL7',
       '133', '408', 'A5', 'Qashqai', 'Corsa', 'Seal', 'Matrix', 'Patrol',
       'TT', 'STI', 'Range Rover Evoque', 'Z4', 'FJ Cruiser', 'ML 320',
       '508', 'Santa Cruz', 'Carens', '1500', 'Pajero Sport', 'Wrangler',
       'MKX', 'E 230', 'J4', 'S-Type', 'Figo', 'Jumpy', 'Granta',
       'Q4 e-tron', 'X5 M', 'Impreza', 'TownAce', 'Cygnet', 'C 180 Coupe',
       'Baleno', 'G 550', 'John Cooper Works', 'WRX', 'MG5', 'D60',
       'Dukato', 'Edge', '93', 'Grand Caravan', 'Caravan', 'Discovery',
       'EC7', 'XE', 'X200', 'Geometry C', 'Amarok', 'Mohave', 'E39',
       'Armada', 'Dokker Van', 'Vitara', '4', 'Touring EV', 'LS 600',
       'GS 350', 'M5', 'Fox RS', 'Trans Sport', 'CC', 'GT86', 'Celica',
       'X-Type', 'Crown Victoria', 'Xsara', 'Impala', 'CX-5', 'Carina',
       'Terios', 'X1', 'Datsun', 'i4', '528', 'Land Cruiser J70',
       'CLA 250', 'Titan', 'Roomster', 'Cami', 'Rosa', 'GC5', 'Shelby',
       'Octavia', 'RX 350', 'Sport Truck Explorer', 'Mach-E', 'Previa',
       'Trooper', 'Fabia', 'Tracker', 'Punto', 'C5', 'T03', '4500',
       'Latitude', 'F0', 'CX-7', 'X35', 'S2000', 'MG 6', 'Touring',
       'NISMO', 'ET5', 'A 220', 'Superb', 'T8', 'C 30', 'Expedition',
       'A 140', 'C 280', 'S 500', 'MG RX8', '-', 'SM 5', 'SLK 200',
       'Quattro', '500C', 'GTO', 'Viano', 'Xterra', 'Traviq', 'A113',
       'CR-Z', 'i8', 'E 63 AMG', 'G37', 'F3R', 'C 240', 'Vito 119',
       'CS95', 'DS3', 'Journey', 'Campo', 'Charade', 'Mondeo', '750',
       '607', 'S3', 'i7', 'MX-3', 'GT', 'GS5', 'Corvette', 'Quoris', 'H2',
       '735', 'i30', 'Ibiza', 'SEL 500', 'A8', 'GLE 400 Coupe', 'Combo',
       '300S', 'DTS', 'Safran', 'T99', 'Range Rover HSE', 'Ciaz',
       'SL 560', 'Skyworth', 'XF', 'Cordoba', 'Supra', 'Cayenne S',
       'Cerato Koup', 'GS3', 'X 250d'])
model_combobox.grid(row=2, column=1, padx=5, pady=5)

year_var = tk.StringVar()
year_combobox = ttk.Combobox(window, textvariable=year_var, values=[1971, 1972, 1973, 1975, 1977, 1980, 1982, 1983, 1984, 1985, 1986,
       1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997,
       1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
       2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
       2020, 2021, 2022, 2023, 2024])
year_combobox.grid(row=3, column=1, padx=5, pady=5)


# year_entry = ttk.Entry(window)
# year_entry.grid(row=3, column=1, padx=5, pady=5)

trans_var = tk.StringVar()
trans_combobox = ttk.Combobox(window, textvariable=trans_var, values=['Automatic', 'Manual'])
trans_combobox.grid(row=4, column=1, padx=5, pady=5)

kilo_var = tk.StringVar()
kilometers_combobox = ttk.Combobox(window, textvariable=kilo_var, values=['40,000 - 49,999', '180,000 - 189,999', '140,000 - 149,999',
       '120,000 - 129,999', '110,000 - 119,999', '20,000 - 29,999',
       '90,000 - 99,999', '0', '200,000', '10,000 - 19,999',
       '50,000 - 59,999', '30,000 - 39,999', '1,000 - 9,999',
       '170,000 - 179,999', '80,000 - 89,999', '70,000 - 79,999',
       '100,000 - 109,999', '60,000 - 69,999', '150,000 - 159,999',
       '130,000 - 139,999', '190,000 - 199,999', '1 - 999',
       '160,000 - 169,999'])
kilometers_combobox.grid(row=5, column=1, padx=5, pady=5)

fuel_var = tk.StringVar()
fuel_combobox = ttk.Combobox(window, textvariable=fuel_var, values=['Hybrid', 'Gasoline', 'Electric', 'Plug-in - Hybrid', 'Diesel',
       'Mild Hybrid'])
fuel_combobox.grid(row=6, column=1, padx=5, pady=5)

engine_size_var = tk.StringVar()
engine_size_entry = ttk.Combobox(window, textvariable=engine_size_var, values=['0 - 499 cc','500 - 999 cc','1,000 - 1,999 cc','2,000 - 2,999 cc', '3,000 - 3,999 cc','4,000 - 4,999 cc','5,000 - 5,999 cc','More than 6,000 cc','Less than 50 kWh','50 - 69 kWh',
        '70 - 89 kWh','90 - 99 kWh', 
         'More than 100 kWh'])
engine_size_entry.grid(row=7, column=1, padx=5, pady=5)

# Create predict button
predict_button = ttk.Button(window, text="Predict", command=predict)
predict_button.grid(row=8, column=1, padx=5, pady=5)

clear_button = ttk.Button(window,text = "Clear", command=clear)
clear_button.grid(row=9, column=1, padx=5, pady=5)


# Run the Tkinter event loop
window.mainloop()
