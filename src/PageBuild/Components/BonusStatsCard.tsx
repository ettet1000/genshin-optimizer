import { CardContent, Divider, ListItem, Typography } from '@mui/material';
import { useContext } from 'react';
import CardLight from '../../Components/Card/CardLight';
import { FieldDisplayList, NodeFieldDisplay } from '../../Components/FieldDisplay';
import { DataContext } from '../../DataContext';
import { input } from '../../Formula';
import { NumNode } from '../../Formula/type';
export default function BonusStatsCard() {
  const { data, character } = useContext(DataContext)
  const bonusStatsKeys = Object.keys(character?.bonusStats)
  if (!bonusStatsKeys.length) return null
  const nodes = bonusStatsKeys.map(k => data.get(input.total[k] as NumNode))
  return <CardLight>
    <CardContent sx={{ py: 1 }}>
      <Typography>Bonus Stats</Typography>
    </CardContent>
    <Divider />
    <CardContent><FieldDisplayList sx={{ my: 0 }} >
      {nodes.map((n, i) => <ListItem key={i}><NodeFieldDisplay node={n} /></ListItem>)}
    </FieldDisplayList></CardContent>
  </CardLight>
}