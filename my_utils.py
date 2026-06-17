
import os
import csv
import numpy as np
from datetime import datetime
from einops.layers.torch import RearrangeMixin

def save_results(args, overall_results, output_dir="output", filename="result.csv"):









    val_mean = np.array(overall_results['best_val_acc']).mean()
    val_std = np.array(overall_results['best_val_acc']).std()
    test_mean = np.array(overall_results['best_test_acc']).mean()
    test_std = np.array(overall_results['best_test_acc']).std()
    val_f1_macro_mean = np.array(overall_results.get('best_val_f1_macro', [])).mean() if overall_results.get('best_val_f1_macro') else None
    val_f1_macro_std = np.array(overall_results.get('best_val_f1_macro', [])).std() if overall_results.get('best_val_f1_macro') else None
    test_f1_macro_mean = np.array(overall_results.get('best_test_f1_macro', [])).mean() if overall_results.get('best_test_f1_macro') else None
    test_f1_macro_std = np.array(overall_results.get('best_test_f1_macro', [])).std() if overall_results.get('best_test_f1_macro') else None


    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)


    file_exists = os.path.exists(csv_path)
    header = [
        "时间戳", "数据集",
        "验证Acc均值", "验证Acc标准差",
        "验证Macro-F1均值", "验证Macro-F1标准差",
        "测试Acc均值", "测试Acc标准差",
        "测试Macro-F1均值", "测试Macro-F1标准差"
    ]
    needs_header = not file_exists
    if file_exists:
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            current_header = next(reader, None)
            needs_header = current_header != header


    with open(csv_path, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)

        if needs_header:
            writer.writerow(header)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            timestamp,
            args.data,
            val_mean,
            val_std,
            val_f1_macro_mean,
            val_f1_macro_std,
            test_mean,
            test_std,
            test_f1_macro_mean,
            test_f1_macro_std
        ])






























def save_results_regression(args, overall_results, output_dir="output", filename="result.csv"):









    val_loss_mean = np.array(overall_results['best_val_loss']).mean()
    val_loss_std = np.array(overall_results['best_val_loss']).std()

    val_rmse_mean = np.array(overall_results['best_val_rmse']).mean()
    val_rmse_std = np.array(overall_results['best_val_rmse']).std()

    val_mae_mean = np.array(overall_results['best_val_mae']).mean()
    val_mae_std = np.array(overall_results['best_val_mae']).std()

    val_r2_mean = np.array(overall_results['best_val_r2']).mean()
    val_r2_std = np.array(overall_results['best_val_r2']).std()

    test_loss_mean = np.array(overall_results['best_test_loss']).mean()
    test_loss_std = np.array(overall_results['best_test_loss']).std()

    test_rmse_mean = np.array(overall_results['best_test_rmse']).mean()
    test_rmse_std = np.array(overall_results['best_test_rmse']).std()

    test_mae_mean = np.array(overall_results['best_test_mae']).mean()
    test_mae_std = np.array(overall_results['best_test_mae']).std()

    test_r2_mean = np.array(overall_results['best_test_r2']).mean()
    test_r2_std = np.array(overall_results['best_test_r2']).std()

    avg_duration = np.array(overall_results['durations']).mean()


    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)


    file_exists = os.path.exists(csv_path)


    with open(csv_path, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            header = [
                "时间戳", "数据集", "平均训练时间",
                "验证Loss均值", "验证Loss标准差",
                "验证RMSE均值", "验证RMSE标准差",
                "验证MAE均值", "验证MAE标准差",
                "验证R2均值", "验证R2标准差",
                "测试Loss均值", "测试Loss标准差",
                "测试RMSE均值", "测试RMSE标准差",
                "测试MAE均值", "测试MAE标准差",
                "测试R2均值", "测试R2标准差"
            ]
            writer.writerow(header)


        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            timestamp,
            args.data,
            f"{avg_duration:.2f}",
            f"{val_loss_mean:.6f}",
            f"{val_loss_std:.6f}",
            f"{val_rmse_mean:.6f}",
            f"{val_rmse_std:.6f}",
            f"{val_mae_mean:.6f}",
            f"{val_mae_std:.6f}",
            f"{val_r2_mean:.6f}",
            f"{val_r2_std:.6f}",
            f"{test_loss_mean:.6f}",
            f"{test_loss_std:.6f}",
            f"{test_rmse_mean:.6f}",
            f"{test_rmse_std:.6f}",
            f"{test_mae_mean:.6f}",
            f"{test_mae_std:.6f}",
            f"{test_r2_mean:.6f}",
            f"{test_r2_std:.6f}"
        ])

    print(f"结果已保存到: {csv_path}")



def save_results_simple(args, overall_results, output_dir="output", filename="result_simple.csv"):




    val_rmse_mean = np.array(overall_results['best_val_rmse']).mean()
    val_rmse_std = np.array(overall_results['best_val_rmse']).std()

    test_rmse_mean = np.array(overall_results['best_test_rmse']).mean()
    test_rmse_std = np.array(overall_results['best_test_rmse']).std()

    test_r2_mean = np.array(overall_results['best_test_r2']).mean()
    test_r2_std = np.array(overall_results['best_test_r2']).std()

    avg_duration = np.array(overall_results['durations']).mean()


    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)


    file_exists = os.path.exists(csv_path)


    with open(csv_path, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            header = [
                "时间戳", "数据集", "训练时间",
                "验证RMSE均值", "验证RMSE标准差",
                "测试RMSE均值", "测试RMSE标准差",
                "测试R2均值", "测试R2标准差"
            ]
            writer.writerow(header)


        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            timestamp,
            args.data,
            f"{avg_duration:.2f}",
            f"{val_rmse_mean:.6f}",
            f"{val_rmse_std:.6f}",
            f"{test_rmse_mean:.6f}",
            f"{test_rmse_std:.6f}",
            f"{test_r2_mean:.6f}",
            f"{test_r2_std:.6f}"
        ])

    print(f"简化结果已保存到: {csv_path}")



























