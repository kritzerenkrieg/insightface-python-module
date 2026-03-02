import argparse
from face_recog.register import register_image
from face_recog.recognize import recognize_image

def main():
    p = argparse.ArgumentParser(description="InsightFace image-based register & recognize")
    sub = p.add_subparsers(dest="cmd")

    r1 = sub.add_parser("register")
    r1.add_argument("--image", required=True)
    r1.add_argument("--name", required=True)
    r1.add_argument("--db", default="./db")

    r2 = sub.add_parser("recognize")
    r2.add_argument("--image", required=True)
    r2.add_argument("--db", default="./db")
    r2.add_argument("--topk", type=int, default=1)
    r2.add_argument("--threshold", type=float, default=0.5)

    args = p.parse_args()
    if args.cmd == "register":
        out = register_image(args.image, args.name, args.db)
        print("Registered:", out)
    elif args.cmd == "recognize":
        res = recognize_image(args.image, args.db, top_k=args.topk, threshold=args.threshold)
        if not res:
            print("No matches")
        else:
            for r in res:
                print(f"{r['name']} (id={r['id']}), score={r['score']:.4f}")
    else:
        p.print_help()

if __name__ == '__main__':
    main()
