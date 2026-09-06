on run
    set appPath to POSIX path of (path to me)
    set projectPath to do shell script "/usr/bin/dirname " & quoted form of appPath
    do shell script "/bin/bash " & quoted form of (projectPath & "/scripts/dev-control.sh") & " stop >/dev/null 2>&1"
end run
