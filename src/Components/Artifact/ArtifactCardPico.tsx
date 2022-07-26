import { Box, Typography } from '@mui/material';
import Assets from '../../Assets/Assets';
import { ArtifactSheet } from '../../Data/Artifacts/ArtifactSheet';
import usePromise from '../../ReactHooks/usePromise';
import { ICachedArtifact } from '../../Types/artifact';
import { allElementsWithPhy, SlotKey } from '../../Types/consts';
import CardDark from '../Card/CardDark';
import SqBadge from '../SqBadge';
import StatIcon, { uncoloredEleIcons } from '../StatIcon';
import ArtifactTooltip from './ArtifactTooltip';

export default function ArtifactCardPico({ artifactObj: art, slotKey: key }: { artifactObj: ICachedArtifact | undefined, slotKey: SlotKey }) {
  const artifactSheet = usePromise(() => art?.setKey && ArtifactSheet.get(art.setKey), [art?.setKey])
  // Blank artifact slot icon
  if (!art || !artifactSheet)
    return <CardDark sx={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <Box sx={{ width: "100%", pb: "100%", position: "relative", }}>
        <Box
          sx={{
            position: "absolute",
            width: "70%", height: "70%",
            left: "50%", top: "50%",
            transform: "translate(-50%, -50%)",
            opacity: 0.7
          }}
          component="img"
          src={Assets.slot[key]}
        />
      </Box>
    </CardDark>

  // Actual artifact icon + info
  const { mainStatKey, rarity, level } = art
  const levelVariant = "roll" + (Math.floor(Math.max(level, 0) / 4) + 1)
  const element = allElementsWithPhy.find(ele => art.mainStatKey.includes(ele))
  const color = element ?? "secondary"

  return <ArtifactTooltip art={art}><CardDark sx={{ display: "flex", flexDirection: "column", position: "relative" }}>
    <Box
      component="img"
      className={`grad-${rarity}star`}
      src={artifactSheet.slotIcons[key]}
      maxWidth="100%"
      maxHeight="100%"
    />
    <Typography sx={{ position: "absolute", m: -0.2, lineHeight: 1, pointerEvents: "none" }} variant="subtitle2"><SqBadge color={levelVariant as any}>+{level}</SqBadge></Typography>
    <Typography variant='h6' sx={{ position: "absolute", bottom: 0, right: 0, lineHeight: 1, }}>
      <SqBadge color={color} sx={{ p: 0.25 }}>{element ? uncoloredEleIcons[element] : StatIcon[mainStatKey]}</SqBadge>
    </Typography>
  </CardDark></ArtifactTooltip>
}
