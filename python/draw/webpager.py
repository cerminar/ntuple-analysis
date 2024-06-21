import os
import shutil
import time
def confirm():
    yesAnswers = ['yes', 'y'];
    noAnswers = ['no', 'n']
    
    answer = input('Yes/No: ').lower().strip()

    if answer in yesAnswers:
        return True
    elif answer in noAnswers:
        return False
    else:
        return confirm()


class WebPageCreator(object):
    def __init__(self, topic_dir, project_dir, base_dir='~/CERNbox/www/plots', tmp_dir='/tmp', samples=None) -> None:
        self.base_dir = base_dir
        self.tmp_path = os.path.join(tmp_dir, project_dir, topic_dir)
        self.project_path = os.path.join(base_dir, project_dir)
        self.topic_path = os.path.join(base_dir, project_dir, topic_dir)
        os.makedirs(self.tmp_path, exist_ok=True)
        self.sample_file = None
        if samples:
            self.sample_file = os.path.join(tmp_dir, project_dir, 'samples.txt')
            f = open(self.sample_file, "w")
            for samp in samples:
                f.write(str(samp)+'\n')
            f.close()
        self.canvases = {}

    def add(self, name, canvas):
        if name in self.canvases.keys():
            print(f'[WebPageCreator]***Warning: overwriting canvas: {name}!')
        self.canvases[name] = canvas
        for ext in ['png', 'pdf']:
            canvas.SaveAs(os.path.join(self.tmp_path, f'{name}.{ext}'))
            time.sleep(0.1)


    def publish(self):
        if len(self.canvases) == 0:
            return        
        if not os.path.exists(self.project_path):
            os.mkdir(self.project_path)
            if os.path.exists(os.path.join(self.base_dir, 'index.php')):
                shutil.copyfile(
                    os.path.join(self.base_dir, 'index.php'), 
                    os.path.join(self.project_path, 'index.php'))
            
        if os.path.exists(self.topic_path):
            print(f'WARNING: directory: {self.topic_path} already exists. Content might be overwritten?')
            if not confirm():
                return
        else:
            os.mkdir(self.topic_path)
            if os.path.exists(os.path.join(self.base_dir, 'index.php')):
                shutil.copyfile(
                    os.path.join(self.base_dir, 'index.php'), 
                    os.path.join(self.topic_path, 'index.php'))

        for name,canvas in self.canvases.items():
            print(f'publishing canvas: {name}')
            for ext in ['png', 'pdf']:
                source = os.path.join(self.tmp_path, f'{name}.{ext}')
                if os.path.exists(source):
                    shutil.copyfile(
                        source, 
                        os.path.join(self.topic_path, f'{name}.{ext}'))
                else:
                    print(f' .  file: {source} missing!')
        
        if self.sample_file:
            self.append_sample_file()

        return


    def append_sample_file(self):
        target_path = os.path.join(self.topic_path, 'samples.txt')

        # Read the content of the target file if it exists
        if os.path.exists(target_path):
            with open(target_path, 'r') as target:
                target_lines = set(target.readlines())
        else:
            target_lines = set()

        # Read the content of the source file
        with open(self.sample_file, 'r') as source:
            source_lines = source.readlines()

        # Filter out the lines that are already in the target file
        new_lines = [line for line in source_lines if line not in target_lines]

        # Append the new lines to the target file
        with open(target_path, 'a') as target:
            target.writelines(new_lines)

        return
    
