import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_basic_stats(df):
    """Базовая статистика по целевым переменным"""
    targets = ['Tube_diameter_nm', 'Tube_length_um', 'Wall_thickness_nm',
               'Pore_density_pores_per_um2', 'Anatase_ratio']

    print("\nСтатистика")
    for col in targets:
        if col in df.columns:
            print(f"{col:30s} mean={df[col].mean():8.2f}  std={df[col].std():8.2f}  "
                  f"min={df[col].min():8.2f}  max={df[col].max():8.2f}")


def plot_distribution(df, column, title):
    if column not in df.columns:
        print(f"Колонка {column} не найдена")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(df[column].dropna(), bins=30, edgecolor='black', alpha=0.8)
    plt.xlabel(column)
    plt.ylabel('Частота')
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_scatter(df, x_col, y_col, title):
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Одна из колонок не найдена: {x_col}, {y_col}")
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(df[x_col], df[y_col], alpha=0.6, s=20)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        print("Недостаточно числовых колонок для корреляции")
        return

    plt.figure(figsize=(10, 8))
    corr = numeric_df.corr()
    plt.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Коэффициент корреляции')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
    plt.title('Матрица корреляций (числовые признаки)')
    plt.tight_layout()
    plt.show()


def main():
    try:
        df = pd.read_csv('dataset.csv')
        print(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
    except FileNotFoundError:
        print("dataset.csv не найден. Сначала запустите data_generator.py")
        return

    plot_basic_stats(df)

    print("\n>>> Нажмите Enter для показа следующего графика...")

    plot_distribution(df, 'Tube_diameter_nm', 'Распределение диаметра нанотрубок')
    input()

    plot_distribution(df, 'Tube_length_um', 'Распределение длины нанотрубок')
    input()

    plot_distribution(df, 'Anatase_ratio', 'Распределение доли анатаза')
    input()

    plot_scatter(df, 'Voltage_V', 'Tube_diameter_nm', 'Напряжение → Диаметр')
    input()

    plot_scatter(df, 'Anodization_time_min', 'Tube_length_um', 'Время → Длина')
    input()

    plot_scatter(df, 'NH4F_wt_percent', 'Pore_density_pores_per_um2', 'NH4F → Плотность пор')
    input()

    plot_scatter(df, 'Annealing_temperature_C', 'Anatase_ratio', 'Температура отжига → Анатаз')
    input()

    plot_correlation_heatmap(df)

    print("\nАнализ завершён")


if __name__ == "__main__":
    main()