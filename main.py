import sys
import argparse
import subprocess
import os

def run_script(script_path, args_list=None):
    """Utility to run sub-scripts using the active virtual environment python interpreter."""
    python_exe = sys.executable
    cmd = [python_exe, script_path]
    if args_list:
        cmd.extend(args_list)
        
    # Inject PYTHONPATH to resolve project imports correctly in child process
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.abspath(__file__))
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = project_root + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = project_root
        
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(
        description="DeepShield: AI-Powered Deepfake Detection Pipeline Orchestrator",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")
    
    # Preprocess command
    subparsers.add_parser("preprocess", help="Run automated face extraction, alignment, and dataset splitting pipeline")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a deepfake detection model")
    train_parser.add_argument("--model", type=str, default="hybrid", choices=["efficientnet_b0", "efficientnet_b3", "xception", "vit", "hybrid"], help="Model architecture")
    train_parser.add_argument("--no-resume", action="store_true", help="Train from scratch rather than resuming from latest checkpoint")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model on the test split and generate metrics/plots")
    eval_parser.add_argument("--model", type=str, default="hybrid", choices=["efficientnet_b0", "efficientnet_b3", "xception", "vit", "hybrid"], help="Model to evaluate")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict deepfake classification on an image or video file")
    predict_parser.add_argument("--input", type=str, required=True, help="Path to input image or video")
    predict_parser.add_argument("--model", type=str, default="hybrid", help="Model checkpoint to load")
    
    # Report command
    subparsers.add_parser("report", help="Generate the Microsoft Word (.docx) final year project report")
    
    # Test command
    subparsers.add_parser("test", help="Execute the unit testing pipeline")

    args, unknown = parser.parse_known_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
        
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.command == "preprocess":
        code = run_script(os.path.join(root_dir, "src", "preprocessing.py"))
        sys.exit(code)
        
    elif args.command == "train":
        train_args = ["--model", args.model]
        if args.no_resume:
            train_args.append("--no-resume")
        code = run_script(os.path.join(root_dir, "src", "train.py"), train_args)
        sys.exit(code)
        
    elif args.command == "evaluate":
        code = run_script(os.path.join(root_dir, "src", "evaluate.py"), ["--model", args.model])
        sys.exit(code)
        
    elif args.command == "predict":
        code = run_script(os.path.join(root_dir, "src", "predict.py"), ["--model", args.model, "--input", args.input])
        sys.exit(code)
        
    elif args.command == "report":
        code = run_script(os.path.join(root_dir, "reports", "generate_doc_report.py"))
        sys.exit(code)
        
    elif args.command == "test":
        # Run unittest directly
        import unittest
        loader = unittest.TestLoader()
        start_dir = os.path.join(root_dir, "tests")
        suite = loader.discover(start_dir)
        runner = unittest.TextTestRunner()
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
