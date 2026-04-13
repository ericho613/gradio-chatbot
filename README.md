Perform the following steps to run the application on a Windows operating system:

1.  In the command prompt, navigate inside the project folder by running the following command: cd [absolute path of the project folder]

2. Create a python virtual environment for the project.  In the terminal, run the following command: python -m venv venv

3. In the terminal, activate the virtual environment by running the following command: venv\Scripts\activate

4. In the terminal, install all required 3rd party packages/dependencies by running the following command: pip3 install -r requirements.txt

5. Create a ".env" file in the root folder/directory.  You can copy the ".env-example" file in the directory, and rename it to ".env".

6. Obtain an API key from OpenAI (https://openai.com/).  Then, in the .env file, create a key named "OPENAI_API_KEY", and assign to it the value of the OpenAI API key using an "=".

7. To start the application, run the following command in the terminal: python main.py

To run the application via docker perform the following steps:

1. Download, install, and run Docker Desktop (https://www.docker.com/products/docker-desktop/) if using a Windows operating system

2. In the command prompt, navigate inside the project folder by running the following command: cd [absolute path of the project folder]

3. To build the docker image and start the container, run "docker-compose up -d" in the terminal

4. To stop and delete the container, run "docker-compose down" in the terminal