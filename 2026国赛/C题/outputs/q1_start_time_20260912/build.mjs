import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const dir='E:/MathModeling/2026国赛/C题/outputs/q1_start_time_20260912';
const data=JSON.parse(await fs.readFile('E:/MathModeling/2026国赛/C题/results/q1_start_time_20260912/workbook_payload.json','utf8'));
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(data.template));
const preview=process.argv.includes('--preview');
if(!preview){
  wb.worksheets.getItem('计划购电量').getRange('B2:B145').values=data.purchases;
  wb.worksheets.getItem('充放电量').getRange('B2:C7').values=data.storage;
  wb.worksheets.getItem('充放电量').getRange('E2:E3').values=data.endpoints;
  wb.recalculate();
  console.log((await wb.inspect({kind:'table',range:'充放电量!A1:E7',include:'values,formulas',tableMaxRows:7,tableMaxCols:5})).ndjson);
  console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20}})).ndjson);
  await (await SpreadsheetFile.exportXlsx(wb)).save(`${dir}/result1.xlsx`);
}
for(const [sheet,range,name] of [['计划购电量','A1:B10','purchase'],['充放电量','A1:E7','storage']]){
  const blob=await wb.render({sheetName:sheet,range,scale:2,format:'png'});
  await fs.writeFile(`${dir}/${preview?'before':'after'}_${name}.png`,new Uint8Array(await blob.arrayBuffer()));
}
