import time
from termcolor import colored

class Logger:
    def __init__(self):
        self.is_debug = False
        self.log_file = None

    def __init__(self, is_debug, log_file):
        self.is_debug = is_debug
        self.log_file = log_file

    def my_colored(self, msg, color):
        if self.log_file is not None:
            return msg
        return colored(msg, color)

    def log(self, msg, critical=False):
        if not self.is_debug and not critical:
            return
        time.ctime()
        msg = self.my_colored(time.strftime('%b %d %Y, %l:%M:%S '), 'blue') + str(msg)
        if critical:
            msg = self.my_colored('[CRITICAL]', 'red') + ' ' + msg
        if self.log_file is None:
            print msg
        else:
            with open(self.log_file, 'w') as log_output:
                log_output.write(msg + '\n')


