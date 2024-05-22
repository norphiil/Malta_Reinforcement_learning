import os

plots_files = os.listdir('plots')

with open('appendix.md', 'w') as f:
    f.write('# Appendix\n\n')
    for file in plots_files:
        f.write('## ' + file.split(".")[0].capitalize() + '\n\n')
        f.write(f'![{file}](plots/{file})\n\n')
