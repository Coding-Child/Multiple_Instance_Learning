import subprocess
import pandas as pd
import json

# 하이퍼파라미터 조합 정의
num_fc_options = [4, 8]
loss_weight_options = [0.01, 0.05]
gamma_options = [0.8, 0.5]

# 실험 결과를 저장할 리스트
results = []

for num_fc in num_fc_options:
    for loss_weight in loss_weight_options:
        for gamma in gamma_options:
            print(f"Experiment {num_fc}, {loss_weight}, {gamma} start")
            # 메인 스크립트 실행 커맨드 생성
            command = f"python run.py --num_fc {num_fc} --loss_weight {loss_weight} --gamma {gamma} --num_epochs {10} --num_train_instance {4} --num_test_instance {8}"
            # subprocess를 사용하여 메인 스크립트 실행
            subprocess.run(command, shell=True)

            # 실행 결과 파일(result.json) 읽기
            with open(f'result_{num_fc}_{loss_weight}_{gamma}.json', 'r') as f:
                result_data = json.load(f)
            print('Load Success Result Data!!')

            # 결과 데이터를 리스트에 추가
            results.append({
                "num_fc": num_fc,
                "loss_weight": loss_weight,
                "gamma": gamma,
                "final_auroc": result_data['final_auroc'],
                "final_f1_score": result_data['final_f1_score'],
                "best_auroc": result_data['best_auroc'],
                "best_f1_score": result_data['best_f1_score']
            })

            print(f"Experiment {num_fc}, {loss_weight}, {gamma} done")

# pandas 데이터프레임으로 결과 데이터 변환
df_results = pd.DataFrame(results)

# 결과 데이터를 엑셀 파일로 저장
excel_path = 'grid_search_results.xlsx'
df_results.to_excel(excel_path, index=False)

print(f"Results saved to {excel_path}")