# import subprocess
#
# # def add_dsflow_env():
# #     shell_command = " source ~/.bash_profile &&  rm -rf ./envs/ &&  mkdir ./envs/ && conda env update -f ./conda.yaml -p ./env"
# #     print('Command ', shell_command)
# #     p2 = subprocess.Popen(shell_command, stdout=subprocess.PIPE, shell=True)
# #     p_status = p2.wait()
# #     return p_status
# #
# # add_dsflow_env()

from setuptools import find_packages,setup


setup(
    name="pulsar_template_ds",
    version="0.1",
    packages=find_packages(),
    description="lib for DS",
    author="Big Apps",
    author_email="zachour@bigapps.fr",
    license='?',
    install_requires=[],
    tests_require=['pytest'],
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
    ],
)
