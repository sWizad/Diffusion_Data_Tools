from waifuc.action import ThreeStageSplitAction
from waifuc.export import SaveExporter
from waifuc.source import LocalSource

source = LocalSource(r'D:\Project\CivitAI\GameArt\hades\training data\game art')
source.attach(
    ThreeStageSplitAction(),
).export(SaveExporter(r'D:\Project\CivitAI\GameArt\hades\draft 2\ingame out'))