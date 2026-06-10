import pexpect
import sys
child = pexpect.spawn('ssh root@rp-f0f296.local "./stream_axi_mt 169.254.118.176 20"', encoding='utf-8')
index = child.expect(['password:', pexpect.EOF, pexpect.TIMEOUT], timeout=3)
if index == 0:
    child.sendline('root')
    child.logfile = sys.stdout
    child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=3)
else:
    print(child.before)
