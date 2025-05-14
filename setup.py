from setuptools import setup, find_packages

# 从 requirements.txt 读取依赖项
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='YuLan-OneSim',
    version='1.0.0', # 您可以根据需要更改版本号
    author='Lei Wang, Heyang Gao, Xiaohe Bo, Xu Chen, Ji-Rong Wen', # 请替换为实际作者信息或团队名称
    author_email='wanglei154@ruc.edu.cn', # 请替换为联系邮箱
    description='YuLan-OneSim: Towards the Next Generation of Social Simulator with Large Language Models',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RUC-GSAI/YuLan-OneSim', # 请替换为您的项目仓库URL
    project_urls={
        'Paper': 'https://arxiv.org/abs/2505.07581',
    },
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    include_package_data=True, # 确保 MANIFEST.in 文件中的数据文件被包含
    install_requires=requirements,
    python_requires='>=3.8', # 根据您的项目实际情况调整
    classifiers=[
        'Development Status :: 3 - Alpha', # 或 Beta, Production/Stable
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', # 假设是 MIT，如果不是请更改
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Sociology',
    ],
    entry_points={
        'console_scripts': [
            'yulan-onesim-cli=main:cli_entry_point', # 指向 src/main.py 中的一个包装函数
            'yulan-onesim-server=app:start_server',  # 指向 src/app.py 中的启动函数
        ],
    },
)

