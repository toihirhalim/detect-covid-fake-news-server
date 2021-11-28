# Detect-covid-fake-news-server
A graphQL django server for scraping, analyse sentiments and detect covid fake news  
the client friendly application is on this [repo](https://github.com/toihirhalim/detect-covid-fake-news) 
which is hosted [here](https://toihirhalim.github.io/detect-covid-fake-news/)  

## how to localy start the server ?
### Clone the project
```
git clone https://github.com/toihirhalim/detect-covid-fake-news-server.git
cd detect-covid-fake-news-server
```
### Create a virtuan environnement
```bash
python3 -m venv venv
```
### Activate the virtual environement
#### On Windows
```bash
venv\Scripts\activate
```
#### On MacOS
```bash
source tvenv/bin/activate
```
### Install requirements
```bash
pip install -r requirements.txt
```
### Start the server
```bash
python3 manage.py runserver
```
## License

[MIT](LICENSE)
