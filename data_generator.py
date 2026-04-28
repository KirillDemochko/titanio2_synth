import numpy as np
import pandas as pd

def generate_dataset(n_samples=500, seed=42, filename="dataset.csv"):
    rng = np.random.default_rng(seed)

    # 1. Входные параметры
    df = pd.DataFrame({
        'Ethylene_glycol_vol_percent': rng.uniform(0, 100, n_samples),
        'Water_vol_percent': rng.uniform(0, 50, n_samples),
        'NH4F_wt_percent': rng.uniform(0.1, 1.0, n_samples),
        'Glycerol_vol_percent': rng.uniform(0, 30, n_samples),
        'HF_vol_percent': rng.uniform(0, 5, n_samples),
        'Voltage_V': rng.uniform(10, 100, n_samples),
        'Anodization_time_min': rng.uniform(10, 480, n_samples),
        'Temperature_anodization_C': rng.uniform(10, 50, n_samples),
        'Annealing_temperature_C': rng.uniform(0, 800, n_samples),
        'Annealing_time_min': rng.uniform(0, 240, n_samples),
        'Substrate_thickness_mm': rng.uniform(0.1, 0.5, n_samples),
        'Substrate_area_cm2': rng.uniform(1, 10, n_samples),
        'Annealing_atmosphere': rng.choice(['Air', 'N2', 'Ar', 'O2', 'None'], n_samples)
    })

    # 2. Физически обоснованные зависимости
    # Диаметр: линейно от напряжения, модуляция фтором и температурой, кросс-эффект V*HF
    df['Tube_diameter_nm'] = (
        15.0 + 1.8 * df['Voltage_V'] + 2.5 * df['HF_vol_percent']
        - 0.4 * df['Temperature_anodization_C'] + 0.05 * df['Ethylene_glycol_vol_percent']
        - 0.02 * df['Voltage_V'] * df['HF_vol_percent']
    )

    # Длина: насыщение по времени, масштабирование вязкостью (EG+Gly), торможение температурой
    visc = 1.0 + 0.015 * df['Ethylene_glycol_vol_percent'] + 0.012 * df['Glycerol_vol_percent']
    temp_inhib = 1.0 / (1.0 + 0.025 * (df['Temperature_anodization_C'] - 15.0))
    df['Tube_length_um'] = (
        5.0 + 40.0 * (1.0 - np.exp(-0.012 * df['Anodization_time_min'])) * visc * temp_inhib
        + 0.15 * df['Water_vol_percent'] - 0.8 * df['HF_vol_percent']
    )

    # Толщина стенки: рост от V, экспоненциальное растворение от HF, квадратичный вклад EG
    df['Wall_thickness_nm'] = (
        8.0 + 0.22 * df['Voltage_V'] - 4.0 * df['HF_vol_percent']
        + 0.0015 * df['Ethylene_glycol_vol_percent']**2
    )

    # Плотность пор: экспонента от NH4F, линейное снижение от T и Gly
    df['Pore_density_pores_per_um2'] = (
        80.0 * np.exp(1.6 * df['NH4F_wt_percent'])
        * (1.0 - 0.008 * df['Temperature_anodization_C'])
        * (1.0 + 0.03 * df['Water_vol_percent'] - 0.005 * df['Glycerol_vol_percent'])
    )

    # Доля анатаза: логистика по T_отжига * насыщение по времени + сдвиг от атмосферы
    atm_map = {'Air': 0.04, 'O2': 0.07, 'N2': -0.02, 'Ar': -0.015, 'None': -0.08}
    atm_eff = df['Annealing_atmosphere'].map(atm_map)
    df['Anatase_ratio'] = (
        (1.0 / (1.0 + np.exp(-0.022 * (df['Annealing_temperature_C'] - 430.0))))
        * (1.0 - np.exp(-0.018 * df['Annealing_time_min'])) + atm_eff
    )

    # 3. Гетероскедастичный шум (ошибка растет с величиной, как в SEM/AFM)
    df['Tube_diameter_nm'] += rng.normal(0, 2.0 + 0.03 * np.abs(df['Tube_diameter_nm']))
    df['Tube_length_um'] += rng.normal(0, 0.4 + 0.04 * np.abs(df['Tube_length_um']))
    df['Wall_thickness_nm'] += rng.normal(0, 0.8 + 0.03 * np.abs(df['Wall_thickness_nm']))
    df['Pore_density_pores_per_um2'] += rng.normal(0, 5.0 + 0.05 * np.abs(df['Pore_density_pores_per_um2']))
    df['Anatase_ratio'] += rng.normal(0, 0.015 + 0.02 * np.abs(df['Anatase_ratio']))

    # 4. Физические границы
    df['Tube_diameter_nm'] = df['Tube_diameter_nm'].clip(10, 200)
    df['Tube_length_um'] = df['Tube_length_um'].clip(0.5, 50)
    df['Wall_thickness_nm'] = df['Wall_thickness_nm'].clip(5, 50)
    df['Pore_density_pores_per_um2'] = df['Pore_density_pores_per_um2'].clip(10, 1000)
    df['Anatase_ratio'] = df['Anatase_ratio'].clip(0, 1)

    df.to_csv(filename, index=False)
    print(f"Генерация завершена: {filename} ({len(df)} строк)")
    return df

if __name__ == "__main__":
    generate_dataset()