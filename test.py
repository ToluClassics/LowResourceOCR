from src.model.model import make_model
from src.data.generator import Tokenizer, DataGenerator
from src.data.evaluation import ocr_metrics
import torch
import yaml
import cv2
import argparse
import torchvision.transforms as T
import numpy as np
import src.data.preprocess as pp

parser = argparse.ArgumentParser(description="Train a language OCR")
parser.add_argument(
    "--lang",
    help="language to train in ",
    required=True,
    choices=["eng", "yor", "igbo"],
)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() == True else "cpu"

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

charset = config[f"{args.lang}_charset"]
tokenizer = Tokenizer(
    charset, lang=args.lang, max_text_length=config[f"{args.lang}_max_text_len"]
)

print("[INFO] Load Trained Checkpoint model")
model = make_model(hidden_dim=512, vocab_len=tokenizer.vocab_size)
model.to(device)
model.load_state_dict(
    torch.load(f"run/checkpoint_weights_{args.lang}_trdg.pt", map_location=device)
)

transform = T.Compose([T.ToTensor()])


def get_memory(model, imgs):
    x = model.conv(model.get_feature(imgs))
    bs, _, H, W = x.shape
    pos = (
        torch.cat(
            [
                model.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                model.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
            ],
            dim=-1,
        )
        .flatten(0, 1)
        .unsqueeze(1)
    )

    return model.transformer.encoder(pos + 0.1 * x.flatten(2).permute(2, 0, 1))


def test(model, test_loader, max_text_length):
    model.eval()
    predicts = []
    gt = []
    imgs = []
    with torch.no_grad():
        for k, batch in enumerate(test_loader):
            src, trg = batch
            imgs.append(src.flatten(0, 1))
            src, trg = src.to(device), trg.to(device)
            memory = get_memory(model, src.float())
            out_indexes = [
                tokenizer.chars.index("SOS"),
            ]
            for i in range(max_text_length):
                mask = model.generate_square_subsequent_mask(i + 1).to(device)
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
                output = model.vocab(
                    model.transformer.decoder(
                        model.query_pos(model.decoder(trg_tensor)),
                        memory,
                        tgt_mask=mask,
                    )
                )
                out_token = output.argmax(2)[-1].item()
                out_indexes.append(out_token)
                if out_token == tokenizer.chars.index("EOS"):
                    break
            predicts.append(tokenizer.decode(out_indexes))
            gt.append(tokenizer.decode(trg.flatten(0, 1)))

            if k == 10:
                break
    return predicts, gt, imgs


dataset = DataGenerator(
    source=config["source"], charset=charset, transform=transform, lang=args.lang
)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=2
)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    print("[INFO] Testing Trained Model")
    predicts, gt, imgs = test(model, test_loader, config[f"{args.lang}_max_text_len"])

    predicts = list(map(lambda x: x.replace("SOS", "").replace("EOS", ""), predicts))
    gt = list(map(lambda x: x.replace("SOS", "").replace("EOS", ""), gt))

    evaluate = ocr_metrics(
        predicts=predicts,
        ground_truth=gt,
    )

    print(
        "Calculate Character Error Rate {}, Word Error Rate {} and Sequence Error Rate {}".format(
            evaluate[0], evaluate[1], evaluate[2]
        )
    )
    for i, item in enumerate(imgs[:10]):
        print("=" * 100, "\n")
        img = item.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("image", pp.adjust_to_see(img))
        # cv2.waitKey(0)
        print("Ground truth:", gt[i])
        print("Prediction :", predicts[i], "\n")
