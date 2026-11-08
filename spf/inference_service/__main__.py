import argparse
from .service import JobManager, load_catalog, make_server


def main():
    parser = argparse.ArgumentParser(description="Historical inference and saved-results playback")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    manager = JobManager(load_catalog(args.catalog), args.output_dir)
    server = None
    try:
        server = make_server(manager, port=args.port)
        print(f"Inference viewer: http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        manager.close()


if __name__ == "__main__":
    main()
