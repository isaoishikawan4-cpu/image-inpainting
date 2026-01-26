"""Local Tkinter application for beard thinning (no network/port required)."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from typing import List, Optional
import threading

from core.inpainting import BeardThinningProcessor
from core.image_utils import (
    resize_image_if_needed,
    merge_masks,
    convert_to_binary_mask
)
import config


class BeardThinningApp:
    """Tkinterベースの髭除去シミュレーションアプリケーション。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("髭除去シミュレーションアプリ - Beard Thinning Simulator")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)

        # 状態管理
        self.original_image: Optional[Image.Image] = None
        self.display_image: Optional[ImageTk.PhotoImage] = None
        self.mask_layers: List[np.ndarray] = []
        self.processor = BeardThinningProcessor()
        self.result_images: List[tuple] = []  # (image, caption)
        self.current_result_index = 0

        # 描画状態
        self.drawing = False
        self.last_x = 0
        self.last_y = 0
        self.brush_size = config.BRUSH_RADIUS_DEFAULT
        self.mask_canvas_image: Optional[Image.Image] = None
        self.mask_draw: Optional[ImageDraw.ImageDraw] = None

        # スケール情報
        self.scale_factor = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0

        self._setup_ui()

    def _setup_ui(self):
        """UIをセットアップする。"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左パネル（入力）
        left_frame = ttk.LabelFrame(main_frame, text="入力", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # 画像アップロードボタン
        upload_btn = ttk.Button(
            left_frame,
            text="画像をアップロード",
            command=self._on_upload_click
        )
        upload_btn.pack(fill=tk.X, pady=(0, 10))

        # キャンバス（画像表示・マスク描画）
        canvas_frame = ttk.Frame(left_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg='gray',
            cursor='cross'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # キャンバスイベント
        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.canvas.bind('<Configure>', self._on_canvas_resize)

        # ブラシサイズスライダー
        brush_frame = ttk.Frame(left_frame)
        brush_frame.pack(fill=tk.X, pady=(10, 5))
        ttk.Label(brush_frame, text="ブラシサイズ:").pack(side=tk.LEFT)
        self.brush_slider = ttk.Scale(
            brush_frame,
            from_=5,
            to=50,
            value=self.brush_size,
            orient=tk.HORIZONTAL,
            command=self._on_brush_size_change
        )
        self.brush_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        self.brush_label = ttk.Label(brush_frame, text=f"{self.brush_size}px")
        self.brush_label.pack(side=tk.LEFT, padx=(5, 0))

        # マスクボタン
        mask_btn_frame = ttk.Frame(left_frame)
        mask_btn_frame.pack(fill=tk.X, pady=5)

        add_mask_btn = ttk.Button(
            mask_btn_frame,
            text="マスクを追加",
            command=self._on_add_mask_click
        )
        add_mask_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        clear_mask_btn = ttk.Button(
            mask_btn_frame,
            text="すべてクリア",
            command=self._on_clear_masks_click
        )
        clear_mask_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # マスクカウンター
        self.mask_counter_var = tk.StringVar(value="0個のマスク")
        mask_counter_label = ttk.Label(left_frame, textvariable=self.mask_counter_var)
        mask_counter_label.pack(pady=5)

        # 薄め具合チェックボックス
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="薄め具合（%）:").pack(anchor=tk.W)

        self.level_vars = {}
        levels_frame = ttk.Frame(left_frame)
        levels_frame.pack(fill=tk.X, pady=5)

        for level in config.DEFAULT_THINNING_LEVELS:
            var = tk.BooleanVar(value=True)
            self.level_vars[level] = var
            cb = ttk.Checkbutton(levels_frame, text=f"{level}%", variable=var)
            cb.pack(side=tk.LEFT, padx=5)

        # 実行ボタン
        self.process_btn = ttk.Button(
            left_frame,
            text="髭薄めを実行",
            command=self._on_process_click
        )
        self.process_btn.pack(fill=tk.X, pady=(10, 0))

        # 右パネル（出力）
        right_frame = ttk.LabelFrame(main_frame, text="結果", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # 結果画像表示キャンバス
        self.result_canvas = tk.Canvas(right_frame, bg='gray')
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        self.result_canvas.bind('<Configure>', self._on_result_canvas_resize)

        # ナビゲーションボタン
        nav_frame = ttk.Frame(right_frame)
        nav_frame.pack(fill=tk.X, pady=(10, 5))

        self.prev_btn = ttk.Button(nav_frame, text="◀ 前へ", command=self._show_prev_result)
        self.prev_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        self.next_btn = ttk.Button(nav_frame, text="次へ ▶", command=self._show_next_result)
        self.next_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # 結果キャプション
        self.result_caption_var = tk.StringVar(value="")
        result_caption_label = ttk.Label(
            right_frame,
            textvariable=self.result_caption_var,
            font=('', 12, 'bold')
        )
        result_caption_label.pack(pady=5)

        # 保存ボタン
        save_btn = ttk.Button(
            right_frame,
            text="結果を保存",
            command=self._on_save_click
        )
        save_btn.pack(fill=tk.X, pady=(5, 0))

        # ステータスバー
        self.status_var = tk.StringVar(value="画像をアップロードして髭の領域をマスクしてください")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # プログレスバー
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(side=tk.BOTTOM, fill=tk.X)

    def _on_upload_click(self):
        """画像アップロードハンドラ。"""
        filetypes = [
            ('画像ファイル', '*.png;*.jpg;*.jpeg;*.webp'),
            ('すべてのファイル', '*.*')
        ]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            try:
                image = Image.open(filepath).convert('RGB')
                self.original_image = resize_image_if_needed(image)
                self._init_mask_canvas()
                self._update_canvas()
                self.status_var.set(f"画像をアップロードしました（サイズ: {self.original_image.size}）")
            except Exception as e:
                messagebox.showerror("エラー", f"画像の読み込みに失敗しました: {str(e)}")

    def _init_mask_canvas(self):
        """マスク描画用のキャンバスを初期化。"""
        if self.original_image:
            self.mask_canvas_image = Image.new('L', self.original_image.size, 0)
            self.mask_draw = ImageDraw.Draw(self.mask_canvas_image)

    def _update_canvas(self):
        """キャンバスを更新。"""
        if self.original_image is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        # スケール計算
        img_width, img_height = self.original_image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.scale_factor = min(scale_x, scale_y, 1.0)

        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)

        # センタリング用オフセット
        self.canvas_offset_x = (canvas_width - new_width) // 2
        self.canvas_offset_y = (canvas_height - new_height) // 2

        # 元画像とマスクを合成
        display = self.original_image.copy()
        if self.mask_canvas_image:
            # マスクを白色で半透明オーバーレイ
            mask_overlay = Image.new('RGBA', display.size, (0, 0, 0, 0))
            mask_array = np.array(self.mask_canvas_image)
            overlay_array = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
            overlay_array[mask_array > 0] = [255, 255, 255, 180]  # 白色半透明
            mask_overlay = Image.fromarray(overlay_array, 'RGBA')
            display = display.convert('RGBA')
            display = Image.alpha_composite(display, mask_overlay)
            display = display.convert('RGB')

        display = display.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.display_image = ImageTk.PhotoImage(display)

        self.canvas.delete('all')
        self.canvas.create_image(
            self.canvas_offset_x,
            self.canvas_offset_y,
            anchor=tk.NW,
            image=self.display_image
        )

    def _on_canvas_resize(self, event):
        """キャンバスリサイズハンドラ。"""
        self._update_canvas()

    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        """キャンバス座標を画像座標に変換。"""
        if self.scale_factor == 0:
            return 0, 0
        img_x = int((canvas_x - self.canvas_offset_x) / self.scale_factor)
        img_y = int((canvas_y - self.canvas_offset_y) / self.scale_factor)
        return img_x, img_y

    def _on_mouse_down(self, event):
        """マウスダウンハンドラ。"""
        if self.original_image is None or self.mask_draw is None:
            return
        self.drawing = True
        self.last_x, self.last_y = self._canvas_to_image_coords(event.x, event.y)

    def _on_mouse_drag(self, event):
        """マウスドラッグハンドラ。"""
        if not self.drawing or self.mask_draw is None:
            return

        x, y = self._canvas_to_image_coords(event.x, event.y)

        # ブラシサイズを考慮した描画
        brush = int(self.brush_size / self.scale_factor) if self.scale_factor > 0 else self.brush_size
        self.mask_draw.line(
            [(self.last_x, self.last_y), (x, y)],
            fill=255,
            width=brush * 2
        )
        self.mask_draw.ellipse(
            [x - brush, y - brush, x + brush, y + brush],
            fill=255
        )

        self.last_x, self.last_y = x, y
        self._update_canvas()

    def _on_mouse_up(self, event):
        """マウスアップハンドラ。"""
        self.drawing = False

    def _on_brush_size_change(self, value):
        """ブラシサイズ変更ハンドラ。"""
        self.brush_size = int(float(value))
        self.brush_label.config(text=f"{self.brush_size}px")

    def _on_add_mask_click(self):
        """マスク追加ハンドラ。"""
        if self.mask_canvas_image is None:
            self.status_var.set("先に画像をアップロードしてください")
            return

        mask_array = np.array(self.mask_canvas_image)
        if np.max(mask_array) == 0:
            self.status_var.set("マスクが描画されていません。髭の領域を白で描画してください。")
            return

        binary_mask = convert_to_binary_mask(mask_array)
        self.mask_layers.append(binary_mask)
        self.mask_counter_var.set(f"{len(self.mask_layers)}個のマスク")
        self.status_var.set(f"マスクを追加しました！ 現在: {len(self.mask_layers)}個のマスク")

    def _on_clear_masks_click(self):
        """マスククリアハンドラ。"""
        self.mask_layers = []
        self._init_mask_canvas()
        self._update_canvas()
        self.mask_counter_var.set("0個のマスク")
        self.status_var.set("すべてのマスクをクリアしました")

    def _on_process_click(self):
        """髭薄め実行ハンドラ。"""
        if self.original_image is None:
            messagebox.showerror("エラー", "画像をアップロードしてください")
            return

        if not self.mask_layers:
            messagebox.showerror("エラー", "髭の領域をマスクしてください")
            return

        selected_levels = [level for level, var in self.level_vars.items() if var.get()]
        if not selected_levels:
            messagebox.showerror("エラー", "少なくとも1つの薄め具合レベルを選択してください")
            return

        # 別スレッドで処理
        self.process_btn.config(state='disabled')
        self.progress['value'] = 0
        self.status_var.set("処理中...")

        thread = threading.Thread(
            target=self._process_thinning,
            args=(selected_levels,)
        )
        thread.start()

    def _process_thinning(self, thinning_levels: List[int]):
        """髭薄め処理を実行（別スレッド）。"""
        try:
            combined_mask = merge_masks(self.mask_layers)

            def progress_callback(current, total, level):
                progress_value = (current / total) * 100
                self.root.after(0, lambda: self.progress.configure(value=progress_value))
                if level == "inpainting":
                    self.root.after(0, lambda: self.status_var.set("髭を除去中..."))
                else:
                    self.root.after(0, lambda: self.status_var.set(f"Level {level}% をブレンド中..."))

            results, messages = self.processor.process_thinning(
                self.original_image,
                combined_mask,
                thinning_levels,
                progress_callback=progress_callback
            )

            if not results:
                self.root.after(0, lambda: messagebox.showerror("エラー", "処理に失敗しました\n" + "\n".join(messages)))
                return

            # 結果を保存
            self.result_images = [(self.original_image, "オリジナル (0% 薄め)")]
            for level in sorted(results.keys()):
                caption = f"{level}% 薄め" if level < 100 else "完全除去 (100%)"
                self.result_images.append((results[level], caption))

            self.current_result_index = 0

            # UIを更新
            self.root.after(0, self._update_result_display)
            self.root.after(0, lambda: self.status_var.set(f"完了！ {len(results)}段階の髭薄め画像を生成しました"))
            self.root.after(0, lambda: self.progress.configure(value=100))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("エラー", f"エラーが発生しました: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.process_btn.config(state='normal'))

    def _update_result_display(self):
        """結果表示を更新。"""
        if not self.result_images:
            return

        image, caption = self.result_images[self.current_result_index]
        self.result_caption_var.set(f"{caption} ({self.current_result_index + 1}/{len(self.result_images)})")

        # キャンバスサイズに合わせてリサイズ
        canvas_width = self.result_canvas.winfo_width()
        canvas_height = self.result_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        img_width, img_height = image.size
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        display = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.result_display_image = ImageTk.PhotoImage(display)

        self.result_canvas.delete('all')
        self.result_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            anchor=tk.CENTER,
            image=self.result_display_image
        )

    def _on_result_canvas_resize(self, event):
        """結果キャンバスリサイズハンドラ。"""
        if self.result_images:
            self._update_result_display()

    def _show_prev_result(self):
        """前の結果を表示。"""
        if self.result_images and self.current_result_index > 0:
            self.current_result_index -= 1
            self._update_result_display()

    def _show_next_result(self):
        """次の結果を表示。"""
        if self.result_images and self.current_result_index < len(self.result_images) - 1:
            self.current_result_index += 1
            self._update_result_display()

    def _on_save_click(self):
        """結果保存ハンドラ。"""
        if not self.result_images:
            messagebox.showinfo("情報", "保存する結果がありません")
            return

        # 保存先フォルダを選択
        folder = filedialog.askdirectory(title="保存先フォルダを選択")
        if not folder:
            return

        try:
            for i, (image, caption) in enumerate(self.result_images):
                # ファイル名を作成
                safe_caption = caption.replace('%', 'pct').replace(' ', '_').replace('/', '_')
                filename = f"{i:02d}_{safe_caption}.png"
                filepath = f"{folder}/{filename}"
                image.save(filepath)

            messagebox.showinfo("完了", f"{len(self.result_images)}枚の画像を保存しました")
            self.status_var.set(f"結果を {folder} に保存しました")
        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました: {str(e)}")


def main():
    """メインエントリーポイント。"""
    root = tk.Tk()
    app = BeardThinningApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
