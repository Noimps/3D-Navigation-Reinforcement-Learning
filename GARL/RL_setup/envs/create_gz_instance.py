import os
import shutil
import re
from pathlib import Path
import time
class GzWorldConfig():
    def __init__(self, world_path, instance_id, default_port=9002):
        # Fix: Assign world_path to an instance variable
        self.original_world_path = os.path.abspath(world_path)
        self.new_port = default_port+ (instance_id* 10)
        self.default_port = default_port
        
        # Original model path is hardcoded as requested
        self.original_model_repo_path = "/home/noimps/garl/ardupilot_gazebo/models/iris_with_lidar"
        self.original_model_name = os.path.basename(self.original_model_repo_path)

        self.copied_model_path = None
        self.copied_world_path = None
        self.new_model_name = None # Will be set after model config

        # Call model configuration
        self.new_model_name, self.copied_model_path = self.config_new_model()

        # Call world configuration only if model config was successful
        if self.new_model_name and self.copied_model_path:
            self.copied_world_path = self.config_new_world(self.new_model_name, self.original_world_path)
        else:
            print("Model configuration failed, skipping world configuration.")


    def config_new_world(self, new_model_name, original_world_path):

        if not os.path.isfile(original_world_path):
            print(f"Error: Original world file not found: {original_world_path}")
            return None
        world_dir = Path(os.path.dirname(original_world_path)).parent
        original_world_filename = os.path.basename(original_world_path)
        base_name, ext = os.path.splitext(original_world_filename)
        new_world_filename = f"{base_name}_{self.new_port}{ext}"
        copied_world_path = os.path.join(world_dir, new_world_filename)
        
        if os.path.exists(copied_world_path):
            return copied_world_path
        else:
            print(f"Copying '{original_world_filename}' to '{copied_world_path}'...")
            try:
                shutil.copyfile(original_world_path, copied_world_path)
            #    print("World copy successful.")
            except Exception as e:
                print(f"Error copying world file: {e}")
                return None

       # print(f"Opening '{copied_world_path}' to replace model reference...")
        try:
            with open(copied_world_path, 'r') as f:
                content = f.read()

            # Regex to find model://<original_model_name> and replace it
            # re.escape() is used to handle potential special characters in the name
            model_ref_pattern = re.compile(r'model://' + re.escape(self.original_model_name))
            new_content = model_ref_pattern.sub(f'model://{new_model_name}', content)

            name_pattern = re.compile(f'<world name=\"{original_world_filename.split(".")[0]}\">')
            new_content = name_pattern.sub(f'<world name=\"{copied_world_path.split("/")[-1].split(".")[0]}\">', new_content)
            if new_content == content:
                print(f"No changes made to {copied_world_path}. Original model reference not found or already updated.")
            else:
                with open(copied_world_path, 'w') as f:
                    f.write(new_content)
            #    print(f"Successfully updated model reference in {copied_world_path}.")

        except Exception as e:
            print(f"Error processing world file: {e}")
            return None

      #  print(f"New world configured at: {copied_world_path}")
        print("--------------------------------------------")

        return copied_world_path

    def config_new_model(self):

        # Fix: Refer to original_model_repo_path here
        if not os.path.isdir(self.original_model_repo_path):
            # Fix: Use the correct path for the error message
            print(f"Error: Original model repository not found: {self.original_model_repo_path}")
            return None, None
        
        models_dir = os.path.dirname(self.original_model_repo_path)
        # Fix: Use original_model_name from self
        original_model_name = self.original_model_name 
        new_model_name = f"{original_model_name}_{self.new_port}"
        copied_model_path = os.path.join(models_dir, new_model_name)
        model_sdf_path = os.path.join(copied_model_path, "model.sdf")

        if os.path.exists(copied_model_path):
            print(f"Warning: Model '{new_model_name}' already exists. Skipping copy.")
            return new_model_name, copied_model_path

        else:
            print(f"Copying '{original_model_name}' to '{copied_model_path}'...")
            try:
                shutil.copytree(self.original_model_repo_path, copied_model_path)
                print("Copy successful.")
            except Exception as e:
                print(f"Error copying model: {e}")
                self.cleanup()
                return None, None

        if not os.path.exists(model_sdf_path):
            print(f"Error: model.sdf not found in the copied model directory: {model_sdf_path}")
            return None, None

        #print(f"Opening '{model_sdf_path}' to replace port {self.default_port} with {self.new_port}...")
        try:
            with open(model_sdf_path, 'r') as f:
                content = f.read()

            # Update fdm_port_in
            port_pattern = re.compile(r'(<fdm_port_in>\s*)' + str(self.default_port) + r'(\s*</fdm_port_in>)')
            new_content = port_pattern.sub(r'\g<1>' + str(self.new_port) + r'\g<2>', content)

            # Update model name
            name_pattern = re.compile(r'(<model name=")' + re.escape(original_model_name) + r'(">)') # Use re.escape here too
            new_content = name_pattern.sub(r'\g<1>' + new_model_name + r'\g<2>', new_content)

            sensor_pattern = re.compile(r'__model__')
            new_content = sensor_pattern.sub(str(self.new_port), new_content)

            if new_content == content:
                print(f"No changes made to {model_sdf_path}. Port {self.default_port} not found or name already updated.")
            else:
                with open(model_sdf_path, 'w') as f:
                    f.write(new_content)
                    
             #   print(f"Successfully configured {model_sdf_path}.")

        except Exception as e:
            print(f"Error processing model.sdf: {e}")
            self.cleanup()
            return None, None

        #print(f"New model configured at: {copied_model_path}")
     #   print("--------------------------------------------")
       # print('\n')
        #time.sleep(3)
        return new_model_name, copied_model_path


    def get_launch_gz(self):
        
        return f"gz sim -r -s {self.copied_world_path.split('/')[-1]}"


    def cleanup(self):
        if self.copied_world_path and os.path.exists(self.copied_world_path):
            print(f"Removing copied world: {self.copied_world_path}...")
            try:
                os.remove(self.copied_world_path)
          #      print("World file removal successful.")
                self.copied_world_path = None
            except Exception as e:
                print(f"Error removing world file: {e}")
        if self.copied_model_path and os.path.exists(self.copied_model_path):
        #   print(f"Removing copied model: {self.copied_model_path}...")
            try:
                shutil.rmtree(self.copied_model_path)
          #      print("Model removal successful.")
                self.copied_model_path = None
                self.new_model_name = None
            except Exception as e:
                print(f"Error removing model: {e}")




#s = GzWorldConfig("/home/noimps/garl/ardupilot_gazebo/worlds/left.sdf", 1)

