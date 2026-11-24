"""
Thin client. One process per question:
  python3 /app/app.py --url START_URL --query-id Q1 --query "..."
Talks to the resident server (server.py) and ALWAYS writes /app/output/<query-id>_output.json.
"""
import argparse, json, os, socket, time

SOCK = "/tmp/agent.sock"
OUT_DIR = "/app/output"


def ask(url, query, budget=28):
    t0 = time.time()
    while time.time() - t0 < budget:          # server may still be loading the model
        c = socket.socket(socket.AF_UNIX)
        try:
            c.settimeout(max(1, budget - (time.time() - t0)))
            c.connect(SOCK)
            c.sendall((json.dumps({"url": url, "query": query}) + "\n").encode())
            return json.loads(c.makefile().readline())
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.5)
        except Exception:
            break
        finally:
            c.close()
    return {"answer": "", "citations": []}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="")
    p.add_argument("--query-id", required=True)
    p.add_argument("--query", required=True)
    a = p.parse_args()

    res = ask(a.url, a.query)
    res.setdefault("answer", "")
    res.setdefault("citations", [])
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{a.query_id}_output.json", "w") as f:
        json.dump(res, f)
