import subprocess

def run_script(script_path):
    try:
        subprocess.run(['python', script_path], check=True)
        print(f"Successfully processed {script_path}")

    except subprocess.CalledProcessError:
        print(f"Failed to process {script_path}")




def run_notebook(notebook_path):
    try:
        subprocess.run([
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--inplace',
            notebook_path
        ], check=True)
        print(f"Successfully processed {notebook_path}")
    except subprocess.CalledProcessError:
        print(f"Failed to process {notebook_path}")


if __name__ == "__main__":
    workflow_steps = [
        ('notebook', '01_data_preparation.ipynb')
        ,('notebook', '02_feature_engineering.ipynb')
        ,('script', '03_train_model.py')
         ,('script', '04_evaluation.py')
    ]

    for step_type, file_name in workflow_steps:
        if step_type == 'script':
            run_script(file_name)
        elif step_type == 'notebook':
            run_notebook(file_name)
