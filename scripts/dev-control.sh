#!/usr/bin/env bash

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$PROJECT_DIR/.run"
# Finder starts application bundles with a minimal PATH, which does not include
# Homebrew's Node installation. Keep this explicit so double-click launch works.
PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
NODE_BIN="$(command -v node 2>/dev/null || true)"
NPM_BIN="$(command -v npm 2>/dev/null || true)"
VITE_ENTRY="$PROJECT_DIR/node_modules/vite/bin/vite.js"
BACKEND_PORT=8000
FRONTEND_PORT=5173
BACKEND_PID_FILE="$RUN_DIR/backend.pid"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"
BACKEND_LOG="$RUN_DIR/backend.log"
FRONTEND_LOG="$RUN_DIR/frontend.log"

mkdir -p "$RUN_DIR"

find_python() {
  local candidate conda_bin

  if [[ -n "${AI_FIRST_PYTHON:-}" && -x "${AI_FIRST_PYTHON}" ]]; then
    echo "$AI_FIRST_PYTHON"
    return 0
  fi

  for candidate in \
    "$HOME/miniconda3/envs/datamining/bin/python" \
    "$HOME/anaconda3/envs/datamining/bin/python" \
    "$HOME/miniforge3/envs/datamining/bin/python" \
    "$HOME/mambaforge/envs/datamining/bin/python"; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  conda_bin="$(command -v conda 2>/dev/null || true)"
  if [[ -n "$conda_bin" ]]; then
    candidate="$("$conda_bin" run -n datamining python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  fi

  command -v python3 2>/dev/null || true
}

PYTHON_BIN="$(find_python)"

port_pids() {
  lsof -nP -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null || true
}

is_project_process() {
  local pid="$1"
  local expected="$2"
  local process_cwd command

  process_cwd="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')"
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [[ "$process_cwd" == "$PROJECT_DIR" && "$command" == *"$expected"* ]]
}

wait_for_url() {
  local url="$1"
  local attempts=0
  while (( attempts < 30 )); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.2
    attempts=$((attempts + 1))
  done
  return 1
}

start_backend() {
  local pid
  pid="$(port_pids "$BACKEND_PORT" | head -n 1)"
  if [[ -n "$pid" ]]; then
    if is_project_process "$pid" "server.py --port $BACKEND_PORT"; then
      echo "$pid" >"$BACKEND_PID_FILE"
      echo "判题服务已经在运行。"
      return 0
    fi
    echo "端口 $BACKEND_PORT 已被其他程序占用，无法启动判题服务。" >&2
    return 1
  fi

  if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "没有找到 Python。请先安装 Python 3.10+ 及 requirements.txt 中的依赖。" >&2
    return 1
  fi

  if ! "$PYTHON_BIN" -c "import numpy, torch" >/dev/null 2>&1; then
    echo "Python 缺少 numpy 或 torch，请执行：$PYTHON_BIN -m pip install -r '$PROJECT_DIR/requirements.txt'" >&2
    return 1
  fi

  cd "$PROJECT_DIR" || return 1
  nohup "$PYTHON_BIN" "$PROJECT_DIR/server.py" --port "$BACKEND_PORT" >>"$BACKEND_LOG" 2>&1 </dev/null &
  pid=$!
  echo "$pid" >"$BACKEND_PID_FILE"

  if ! wait_for_url "http://127.0.0.1:$BACKEND_PORT/api/problems"; then
    echo "判题服务启动失败，请查看：$BACKEND_LOG" >&2
    return 1
  fi
  echo "判题服务已启动。"
}

start_frontend() {
  local pid
  pid="$(port_pids "$FRONTEND_PORT" | head -n 1)"
  if [[ -n "$pid" ]]; then
    if is_project_process "$pid" "node_modules/vite/bin/vite.js"; then
      echo "$pid" >"$FRONTEND_PID_FILE"
      echo "网页服务已经在运行。"
      return 0
    fi
    echo "端口 $FRONTEND_PORT 已被其他程序占用，无法启动网页。" >&2
    return 1
  fi

  if [[ -z "$NODE_BIN" || ! -x "$NODE_BIN" ]]; then
    echo "没有找到 Node.js。请安装 Node.js 后再启动网站。" >&2
    return 1
  fi

  if [[ ! -f "$VITE_ENTRY" ]]; then
    echo "首次运行，正在安装网页依赖..."
    if [[ -z "$NPM_BIN" || ! -x "$NPM_BIN" ]]; then
      echo "没有找到 npm，无法安装网页依赖。" >&2
      return 1
    fi
    cd "$PROJECT_DIR" || return 1
    "$NPM_BIN" install || return 1
  fi

  cd "$PROJECT_DIR" || return 1
  nohup "$NODE_BIN" "$VITE_ENTRY" --host 127.0.0.1 --port "$FRONTEND_PORT" >>"$FRONTEND_LOG" 2>&1 </dev/null &
  pid=$!
  echo "$pid" >"$FRONTEND_PID_FILE"

  if ! wait_for_url "http://127.0.0.1:$FRONTEND_PORT/"; then
    echo "网页服务启动失败，请查看：$FRONTEND_LOG" >&2
    return 1
  fi
  echo "网页服务已启动。"
}

stop_service() {
  local label="$1"
  local port="$2"
  local expected="$3"
  local pid stopped=0

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    if is_project_process "$pid" "$expected"; then
      kill "$pid" 2>/dev/null || true
      stopped=1
    else
      echo "$label 使用的端口 $port 被其他程序占用，未做处理。" >&2
    fi
  done < <(port_pids "$port")

  if (( stopped == 1 )); then
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      [[ -z "$(port_pids "$port")" ]] && break
      sleep 0.2
    done
    echo "$label 已停止。"
  else
    echo "$label 当前没有运行。"
  fi
}

show_status() {
  local backend_pid frontend_pid
  backend_pid="$(port_pids "$BACKEND_PORT" | head -n 1)"
  frontend_pid="$(port_pids "$FRONTEND_PORT" | head -n 1)"
  [[ -n "$backend_pid" ]] && echo "判题服务：运行中（PID ${backend_pid}）" || echo "判题服务：未运行"
  [[ -n "$frontend_pid" ]] && echo "网页服务：运行中（PID ${frontend_pid}）" || echo "网页服务：未运行"
}

case "${1:-}" in
  start)
    start_backend || exit 1
    start_frontend || exit 1
    echo "网站地址：http://127.0.0.1:$FRONTEND_PORT/"
    if [[ "${NO_OPEN:-0}" != "1" ]]; then
      open "http://127.0.0.1:$FRONTEND_PORT/"
    fi
    ;;
  stop)
    stop_service "网页服务" "$FRONTEND_PORT" "node_modules/vite/bin/vite.js"
    stop_service "判题服务" "$BACKEND_PORT" "server.py --port $BACKEND_PORT"
    rm -f "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  status)
    show_status
    ;;
  *)
    echo "用法：$0 {start|stop|restart|status}" >&2
    exit 1
    ;;
esac
