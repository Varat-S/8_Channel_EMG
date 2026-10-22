from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg.protocol import FRAME_SIZE, FrameDecoder, decode_frame


CAPTURED_FRAME = bytes.fromhex(
    "55 66 26 e1 "
    "ff 83 ec 04 ff 83 26 10 ff 80 00 00 ff 82 c5 66 "
    "ff 80 00 00 ff 80 00 00 ff 80 00 00 ff 80 00 00 "
    "66 55"
)


class ProtocolTests(unittest.TestCase):
    def test_captured_frame_layout(self) -> None:
        self.assertEqual(len(CAPTURED_FRAME), FRAME_SIZE)
        frame = decode_frame(CAPTURED_FRAME)
        self.assertEqual(frame.metadata_word, 0x26E1)
        self.assertEqual(len(frame.channel_counts), 8)
        self.assertEqual(frame.channel_counts[0], -8_131_580)
        self.assertEqual(frame.channel_counts[1], -8_182_256)
        self.assertEqual(frame.channel_counts[2], -8_388_608)

    def test_incremental_decoder_handles_every_split_point(self) -> None:
        for split in range(len(CAPTURED_FRAME) + 1):
            decoder = FrameDecoder()
            frames = decoder.feed(CAPTURED_FRAME[:split])
            frames += decoder.feed(CAPTURED_FRAME[split:])
            self.assertEqual(len(frames), 1, f"failed at split {split}")
            self.assertEqual(frames[0].raw, CAPTURED_FRAME)

    def test_decoder_resynchronizes_after_noise_and_bad_candidate(self) -> None:
        decoder = FrameDecoder()
        malformed = b"\x55\x66" + (b"\x00" * (FRAME_SIZE - 2))
        frames = decoder.feed(b"noise" + malformed + CAPTURED_FRAME)
        self.assertEqual([frame.raw for frame in frames], [CAPTURED_FRAME])
        self.assertGreater(decoder.discarded_bytes, 0)
        self.assertEqual(decoder.bad_candidates, 1)


if __name__ == "__main__":
    unittest.main()
