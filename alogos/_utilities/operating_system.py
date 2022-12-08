import os as _os
import re as _re


NEWLINE = _os.linesep


def ensure_file_extension(filepath, extension):
    """Ensure that a filepath ends with a given extension.

    References
    ----------
    - https://docs.python.org/3.8/library/os.path.html#os.path.splitext

    """
    # Precondition
    if not filepath:
        raise ValueError('Invalid filepath: "{}"'.format(filepath))
    if not extension:
        raise ValueError('Invalid extension: "{}"'.format(filepath))
    if not extension.startswith("."):
        extension = "." + extension

    # Transformation
    if not filepath.endswith(extension):
        filepath += extension

    # Postcondition
    if not filepath or not isinstance(filepath, str):
        raise ValueError('Could not create a valid filepath: "{}"'.format(filepath))
    return filepath


def delete_file(filepath):
    """Delete a file.

    References
    ----------
    - https://docs.python.org/3/library/os.html#os.remove

    """
    _os.remove(filepath)


def ensure_new_path(path):
    """Check whether a given path exists. If yes, try to find a novel, incremented variant of it.

    Caution: Not threadsafe.

    Examples
    --------
    - If ``some_dir`` exists, it becomes ``some_dir_1.mp3``, then ``some_dir_2.mp3``, ...
    - If ``some_file`` exists, it becomes ``some_file_1``, then ``some_file_2``, ...
    - If ``some_file.mp3`` exists, it becomes ``some_file_1.mp3``, then ``some_file_2.mp3``, ...
    - If ``x_1.y_2.z.txt`` exists, it becomes ``x_1.y_2.z_1.txt``, then ``x_1.y_2.z_2.txt``, ...

    """

    def parse_type1(string):
        try:
            match = _re.search(r"(?P<base>.*?_)(?P<num>\d+)(?P<ext>\.[^\.]+)$", string)
            result = match.group("base"), match.group("num"), match.group("ext")
        except Exception:
            result = None
        return result

    def parse_type2(string):
        try:
            match = _re.search(r"(?P<base>.*?)(?P<ext>\.[^\.]+)$", string)
            result = match.group("base"), match.group("ext")
        except Exception:
            result = None
        return result

    def parse_type3(string):
        try:
            match = _re.search(r"(?P<base>.*?_)(?P<num>\d+)$", string)
            result = match.group("base"), match.group("num")
        except Exception:
            result = None
        return result

    while _os.path.exists(path):
        res1 = parse_type1(path)
        res2 = parse_type2(path)
        res3 = parse_type3(path)
        if res1 is not None:
            base, num, ext = res1
            path = base + str(int(num) + 1) + ext
        elif res2 is not None:
            base, ext = res2
            path = base + "_1" + ext
        elif res3 is not None:
            base, num = res3
            path = base + str(int(num) + 1)
        else:
            path = path + "_1"
    return path


def create_directory(dirpath):
    """Given a directory path, create it and all necessary parent directories.

    Do nothing if the directory already exists.

    References
    ----------
    - https://docs.python.org/2/library/os.html#os.makedirs

    """
    _os.makedirs(dirpath, exist_ok=True)
    return dirpath


def open_in_webbrowser(
    html_text, start_delay=0.1, stop_delay_fast=0.25, stop_delay_slow=10.0
):
    """Open the given HTML text in the default webbrowser.

    A short-lived local HTTP server is created that serves HTML text, so that
    it can be opened in the default webbrowser. An alternative would be to write
    the HTML text to a temporary file and open this file in the webbrowser.

    Parameters
    ----------
    html_text : str
    start_delay : float
        Delay between starting the server and opening the webbrowser in another thread.
        If too short, the webbrowser might send the request before the server is running.
    stop_delay_fast : float
        Delay between opening the webbrowser (or a new tab) and shutting down the server
        after it received the first GET request from the browser.
    stop_delay_slow : float
        Delay between opening the webbrowser (or a new tab) and shutting down the server
        even if it never receives a GET request, e.g. because the browser does not open.

    References
    ----------
    - https://stackoverflow.com/questions/19040055/how-do-i-shutdown-an-httpserver-from-inside-a-request-handler-in-python
    - https://stackoverflow.com/questions/3389305/how-to-silent-quiet-httpserver-and-basichttprequesthandlers-stderr-output
    - https://en.wikipedia.org/wiki/List_of_TCP_and_UDP_port_numbers#Note

    """
    import http.server
    import random
    import threading
    import webbrowser

    def open_browser_and_stop_server():
        # Open webbrowser
        webbrowser.open(url)
        # Start thread 2: Stop the server slowly, even if no GET request is ever received
        timer_stop_slow.start()

    def stop_all():
        # Stop server
        server.shutdown()
        # Stop all timer threads
        timer_stop_slow.cancel()
        timer_stop_fast.cancel()

    class MyHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            """Serve the given HTML text to a GET request from the webbrowser."""
            self.send_response(200, "OK")
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes(html_text, "utf-8"))
            # Start thread 3: Stop the server quickly after the first GET request was received
            if not timer_stop_fast.is_alive():
                timer_stop_fast.start()

        def log_message(self, format, *args):
            """Show no log messages."""

    local_host = "127.0.0.1"
    for _ in range(13):
        try:
            random_port = random.randint(
                49152, 65535
            )  # port range used for "temporary purposes"
            server_address = (local_host, random_port)
            server = http.server.HTTPServer(server_address, MyHandler)
            url = "http://{}:{}".format(local_host, random_port)
            break
        except Exception:
            pass
    else:
        message = (
            "Could not start a local webserver for serving HTML text to a webbrowser. "
            "No free port could be found."
        )
        raise OSError(message)

    # Define threads 1 to 3
    timer_run = threading.Timer(start_delay, open_browser_and_stop_server)
    timer_stop_slow = threading.Timer(stop_delay_slow, stop_all)
    timer_stop_fast = threading.Timer(stop_delay_fast, stop_all)

    # Start thread 1: Open webbrowser with some delay, stop the server with some delay
    timer_run.start()

    # Go into server loop
    server.serve_forever()
