import signal
from contextlib import contextmanager
import requests

DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor": "Google"}

def make_request(url, headers):
    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx and 5xx)
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
    else:
        print(f"Request was successful: {response.status_code}")

def _request_handler(headers):
    def _handler(signum, frame):
        headers['Authorization'] = headers['Authorization'].strip()
        make_request(KEEPALIVE_URL, headers)
    return _handler

@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)

def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    with active_session(delay, interval):
        yield from iterable

# Usage:
# with active_session():
#     # Your long-running code here
#     train_net(n_epochs)
